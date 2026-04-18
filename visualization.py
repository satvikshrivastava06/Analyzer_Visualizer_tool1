# visualization.py — Data Analyzer & Visualizer Tool Advanced Visualization Engine
# Drop next to app.py. Import with: from visualization import render_viz_tab
#
# Chart catalogue (20+ types):
#   STANDARD   : Animated Bar, Multi-Line, Area Stack, Scatter (sized+colored)
#   STATISTICAL: Violin+Box hybrid, Histogram+KDE, Correlation heatmap, Q-Q plot
#   COMPOSITION: Treemap, Sunburst, Waterfall, Funnel, Pie/Donut
#   ADVANCED   : Bubble, Candlestick, Parallel Coords, Radar, Sankey, Density contour
#   ECHARTS    : Animated gauge, Gradient bar race, Calendar heatmap, Liquid fill

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Optional

try:
    from streamlit_echarts import st_echarts
    ECHARTS = True
except ImportError:
    ECHARTS = False

# ─── Design tokens (must match ui_theme.py) ──────────────────────────────────
BG0      = "#0D0F14"
BG1      = "#13151C"
BG2      = "#1A1D28"
BG3      = "#21253A"
BORDER   = "#2A2E45"
ACCENT   = "#6C63FF"
TEXT1    = "#F0F2FF"
TEXT2    = "#8B90B4"
TEXT3    = "#4B5073"
SUCCESS  = "#00D4A0"
WARNING  = "#FFB547"
DANGER   = "#FF5F5F"
INFO     = "#38B6FF"

PALETTE = [
    "#6C63FF","#00D4A0","#FF6B6B","#FFB547",
    "#38B6FF","#C77DFF","#4AE3B5","#FF8C69",
    "#A8DADC","#E63946","#2EC4B6","#FFBF69",
]

SEQUENTIAL = [[0,"#0D0F14"],[0.25,ACCENT],[0.6,SUCCESS],[1,"#C77DFF"]]
SEQ_WARM   = [[0,"#0D0F14"],[0.4,WARNING],[0.8,DANGER],[1,"#FFE0B2"]]
DIVERGING  = [[0,DANGER],[0.5,BG2],[1,SUCCESS]]

def _rgba(hex_color, opacity) -> str:
    """Safely convert hex to rgba string for Plotly stability."""
    hc = hex_color.lstrip('#')
    if len(hc) == 3:
        hc = "".join([c*2 for c in hc])
    r, g, b = (int(hc[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{opacity})"

def _base_layout(**extra):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font         =dict(family="DM Sans, sans-serif", size=12, color=TEXT2),
        title        =dict(font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT1), x=0.02),
        legend       =dict(bgcolor="rgba(26,29,40,0.95)", bordercolor=BORDER,
                           borderwidth=1, font=dict(color=TEXT2, size=11)),
        xaxis        =dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                           tickfont=dict(color=TEXT3, size=11)),
        yaxis        =dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False,
                           tickfont=dict(color=TEXT3, size=11)),
        margin       =dict(l=16, r=16, t=52, b=16),
        colorway     =PALETTE,
        hoverlabel   =dict(bgcolor=BG2, bordercolor=BORDER,
                           font=dict(color=TEXT1, size=12)),
        hovermode    ="closest",
    )
    layout.update(extra)
    return layout


def _apply(fig, title="", **extra):
    config = _base_layout(**extra)
    if title:
        config['title']['text'] = title
    fig.update_layout(**config)
    return fig


def _show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Animated bar (sorted, gradient-colored) ───────────────────────────────
def chart_bar_gradient(df, x, y, color_col=None, title=""):
    if x == y:
        st.warning("Select different columns for Category and Metric to see meaningful insights.")
        return
    agg = df.groupby(x)[y].sum().reset_index().sort_values(y, ascending=True)
    norm = (agg[y] - agg[y].min()) / (agg[y].max() - agg[y].min() + 1e-9)
    colors = [
        f"rgba({int(108+147*v)},{int(99+213*v)},{int(255-55*v)},0.85)"
        for v in norm
    ]
    fig = go.Figure(go.Bar(
        x=agg[y], y=agg[x].astype(str), orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=agg[y].round(1), textposition="outside",
        textfont=dict(color=TEXT2, size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{y}: %{{x:,.1f}}<extra></extra>",
    ))
    _apply(fig, title=title or f"{y} by {x}",
           xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zeroline=False, tickfont=dict(color=TEXT3)),
           yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)",
                      tickfont=dict(color=TEXT1, size=12)))
    _show(fig)


# ── 2. Multi-line with markers & fill ───────────────────────────────────────
def chart_multiline(df, x, y_cols: list, title=""):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = PALETTE[i % len(PALETTE)]
        tmp = df[[x, col]].dropna().sort_values(x)
        fig.add_trace(go.Scatter(
            x=tmp[x], y=tmp[col], name=col, mode="lines+markers",
            line=dict(color=c, width=2.5),
            marker=dict(color=c, size=5, line=dict(color=BG0, width=1.5)),
            fill="tozeroy" if len(y_cols) == 1 else "none",
            fillcolor=_rgba(c, 0.1),
            hovertemplate=f"<b>{col}</b>: %{{y:,.2f}}<extra></extra>",
        ))
    _apply(fig, title=title or f"{', '.join(y_cols)} over {x}",
           hovermode="x unified")
    _show(fig)


# ── 3. Area stack ────────────────────────────────────────────────────────────
def chart_area_stack(df, x, y_cols: list, title=""):
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        c = PALETTE[i % len(PALETTE)]
        tmp = df[[x, col]].dropna().sort_values(x)
        fig.add_trace(go.Scatter(
            x=tmp[x], y=tmp[col], name=col,
            mode="lines", stackgroup="one",
            line=dict(color=c, width=1.5),
            fillcolor=_rgba(c, 0.35),
            hovertemplate=f"<b>{col}</b>: %{{y:,.2f}}<extra></extra>",
        ))
    _apply(fig, title=title or f"Stacked area — {', '.join(y_cols)}", hovermode="x unified")
    _show(fig)


# ── 4. Bubble chart ──────────────────────────────────────────────────────────
def chart_bubble(df, x, y, size_col, color_col=None, title=""):
    relevant = [x, y, size_col] + ([color_col] if color_col else [])
    # Unique columns only for selection to avoid duplicate name DataFrame issues
    selection = []
    for c in relevant:
        if c not in selection: selection.append(c)
    
    tmp = df[selection].dropna().copy()
    s_vals = tmp[size_col]
    if hasattr(s_vals, 'iloc') and len(s_vals.shape) > 1: s_vals = s_vals.iloc[:, 0]

    size_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-9)
    sizes = (size_norm * 50 + 6).tolist()

    if color_col and color_col in tmp.columns:
        cats = tmp[color_col].astype(str).unique().tolist()
        cat_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}
        colors = [cat_map[v] for v in tmp[color_col].astype(str)]
        traces = {}
        for cat in cats:
            mask = tmp[color_col].astype(str) == cat
            traces[cat] = go.Scatter(
                x=tmp[mask][x], y=tmp[mask][y], name=cat, mode="markers",
                marker=dict(size=[sizes[i] for i, m in enumerate(mask) if m],
                            color=cat_map[cat], opacity=0.8,
                            line=dict(color=BG0, width=1)),
                hovertemplate=f"<b>{cat}</b><br>{x}: %{{x}}<br>{y}: %{{y}}<br>{size_col}: %{{marker.size:.1f}}<extra></extra>",
            )
        fig = go.Figure(list(traces.values()))
    else:
        fig = go.Figure(go.Scatter(
            x=tmp[x], y=tmp[y], mode="markers",
            marker=dict(size=sizes, color=sizes, colorscale=SEQUENTIAL,
                        opacity=0.8, showscale=True,
                        colorbar=dict(title=size_col, tickfont=dict(color=TEXT2)),
                        line=dict(color=BG0, width=1)),
            hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<br>{size_col}: %{{marker.size:.1f}}<extra></extra>",
        ))
    _apply(fig, title=title or f"Bubble: {x} vs {y} (size={size_col})")
    _show(fig)


# ── 5. Violin + box hybrid ───────────────────────────────────────────────────
def chart_violin_box(df, x, y, title=""):
    cats = df[x].dropna().unique().tolist()[:12]
    fig = go.Figure()
    for i, cat in enumerate(cats):
        c = PALETTE[i % len(PALETTE)]
        vals = df[df[x] == cat][y].dropna()
        fig.add_trace(go.Violin(
            x=[str(cat)] * len(vals), y=vals, name=str(cat),
            box_visible=True, meanline_visible=True,
            fillcolor=_rgba(c, 0.25), line_color=c,
            points="outliers",
            marker=dict(color=c, size=3, opacity=0.6),
            hovertemplate=f"<b>{cat}</b><br>%{{y:.2f}}<extra></extra>",
        ))
    _apply(fig, title=title or f"{y} distribution by {x}", showlegend=False,
           violingap=0.15, violingroupgap=0.05)
    _show(fig)


# ── 6. Histogram + KDE overlay ───────────────────────────────────────────────
def chart_histogram_kde(df, col, nbins=40, title=""):
    vals = df[col].dropna()
    fig  = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=nbins, name="Count",
        marker=dict(color=ACCENT, opacity=0.65, line=dict(color=BG0, width=0.5)),
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    # KDE via numpy
    from numpy import linspace
    kde_x = linspace(vals.min(), vals.max(), 300)
    h = 1.06 * vals.std() * len(vals)**(-1/5)
    kde_y = np.array([np.sum(np.exp(-0.5*((x - vals.values)/h)**2)) / (len(vals)*h*np.sqrt(2*np.pi)) for x in kde_x])
    # Scale KDE to histogram height
    bin_w = (vals.max() - vals.min()) / nbins
    scale = len(vals) * bin_w
    fig.add_trace(go.Scatter(
        x=kde_x, y=kde_y * scale, name="Density",
        mode="lines", line=dict(color=SUCCESS, width=2.5),
        fill="tozeroy", fillcolor=_rgba(SUCCESS, 0.1),
        hovertemplate="x: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
    ))
    mean, median, std = vals.mean(), vals.median(), vals.std()
    for val, label, color in [(mean,"Mean",WARNING),(median,"Median",INFO)]:
        fig.add_vline(x=val, line_color=color, line_dash="dash", line_width=1.5,
                      annotation_text=f"{label}: {val:.2f}",
                      annotation_font_color=color, annotation_position="top right")
    _apply(fig, title=title or f"Distribution: {col}",
           barmode="overlay",
           annotations=[dict(x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                             text=f"μ={mean:.2f}  σ={std:.2f}  n={len(vals):,}",
                             font=dict(color=TEXT2, size=11), align="right")])
    _show(fig)


# ── 7. Correlation heatmap (annotated) ───────────────────────────────────────
def chart_correlation(df, title=""):
    num = df.select_dtypes(include=[np.number]).dropna(axis=1)
    if len(num.columns) < 2:
        st.warning("Need at least 2 numerical columns for correlation.")
        return
    corr = num.corr()
    mask_upper = np.triu(np.ones_like(corr), k=1).astype(bool)
    z = corr.values.copy()
    z[mask_upper] = None  # show lower triangle only

    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=DIVERGING, zmin=-1, zmax=1,
        text=np.where(~np.isnan(z), np.round(z, 2), ""),
        texttemplate="%{text}", textfont=dict(size=11, color=TEXT1),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", tickfont=dict(color=TEXT2),
                      tickvals=[-1,-0.5,0,0.5,1]),
    ))
    n = len(corr.columns)
    h = max(380, n * 46)
    _apply(fig, title=title or "Correlation matrix",
           height=h, xaxis=dict(side="bottom", tickangle=-30, gridcolor="rgba(0,0,0,0)"),
           yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    _show(fig)


# ── 8. Scatter matrix (SPLOM) ────────────────────────────────────────────────
def chart_splom(df, cols: list, color_col=None, title=""):
    tmp = df[cols + ([color_col] if color_col else [])].dropna()
    dims = [dict(label=c, values=tmp[c]) for c in cols]
    color = tmp[color_col] if color_col else None
    color_kw = dict(color=color, colorscale=SEQUENTIAL,
                    colorbar=dict(title=color_col)) if color_col else dict(color=ACCENT)
    fig = go.Figure(go.Splom(
        dimensions=dims,
        marker=dict(size=4, opacity=0.6, line=dict(width=0), **color_kw),
        diagonal_visible=False,
        showupperhalf=False,
    ))
    _apply(fig, title=title or "Scatter matrix",
           height=600,
           xaxis=dict(showgrid=True, gridcolor=BORDER),
           yaxis=dict(showgrid=True, gridcolor=BORDER))
    _show(fig)


# ── 9. Waterfall ─────────────────────────────────────────────────────────────
def chart_waterfall(df, category_col, value_col, title=""):
    tmp = df.groupby(category_col)[value_col].sum().reset_index().sort_values(value_col)
    running = 0
    measures, colors = [], []
    for v in tmp[value_col]:
        measures.append("relative")
        colors.append(SUCCESS if v >= 0 else DANGER)
    fig = go.Figure(go.Waterfall(
        x=tmp[category_col].astype(str).tolist(),
        y=tmp[value_col].tolist(),
        measure=measures,
        text=[f"{v:+,.0f}" for v in tmp[value_col]],
        textposition="outside", textfont=dict(color=TEXT2, size=11),
        connector=dict(line=dict(color=BORDER, width=1)),
        increasing=dict(marker=dict(color=SUCCESS)),
        decreasing=dict(marker=dict(color=DANGER)),
        totals   =dict(marker=dict(color=ACCENT)),
        hovertemplate="<b>%{x}</b><br>Value: %{y:+,.2f}<extra></extra>",
    ))
    _apply(fig, title=title or f"Waterfall: {value_col} by {category_col}")
    _show(fig)


# ── 10. Funnel ───────────────────────────────────────────────────────────────
def chart_funnel(df, stage_col, value_col, title=""):
    tmp = df.groupby(stage_col)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
    fig = go.Figure(go.Funnel(
        y=tmp[stage_col].astype(str).tolist(),
        x=tmp[value_col].tolist(),
        textinfo="value+percent total",
        textfont=dict(color=TEXT1, size=12),
        marker=dict(
            color=[PALETTE[i % len(PALETTE)] for i in range(len(tmp))],
            line=dict(width=1, color=BG0),
        ),
        connector=dict(fillcolor=f"{BORDER}"),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} (%{percentTotal:.1%})<extra></extra>",
    ))
    _apply(fig, title=title or f"Funnel: {stage_col}")
    _show(fig)


# ── 11. Treemap (custom colors, breadcrumb) ───────────────────────────────────
def chart_treemap(df, path_cols: list, value_col, title=""):
    # Ensure value_col is not in path_cols to keep it numeric
    paths = [p for p in path_cols if p != value_col]
    if not paths: paths = path_cols[:1]
    
    tmp = df.dropna(subset=paths + [value_col]).copy()
    for c in paths: tmp[c] = tmp[c].astype(str)
    
    fig = px.treemap(tmp, path=paths, values=value_col,
                     color=value_col, color_continuous_scale=SEQUENTIAL,
                     title=title or f"Treemap: {' > '.join(paths)}")
    fig.update_traces(
        textfont=dict(size=13, color=TEXT1),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>Share: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color=BG0, width=2)),
    )
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=TEXT2)))
    _apply(fig, height=500)
    _show(fig)


# ── 12. Sunburst (multi-level donut) ─────────────────────────────────────────
def chart_sunburst(df, path_cols: list, value_col, title=""):
    paths = [p for p in path_cols if p != value_col]
    if not paths: paths = path_cols[:1]

    tmp = df.dropna(subset=paths + [value_col]).copy()
    for c in paths: tmp[c] = tmp[c].astype(str)
    
    fig = px.sunburst(tmp, path=paths, values=value_col,
                      color=value_col, color_continuous_scale=SEQUENTIAL)
    fig.update_traces(
        textfont=dict(size=12, color=TEXT1),
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.2f}<br>Share: %{percentParent:.1%}<extra></extra>",
        marker=dict(line=dict(color=BG0, width=1.5)),
        insidetextorientation="radial",
        leaf=dict(opacity=0.9),
    )
    _apply(fig, title=title or f"Sunburst: {' > '.join(path_cols)}", height=520)
    fig.update_coloraxes(colorbar=dict(tickfont=dict(color=TEXT2)))
    _show(fig)


# ── 13. Donut / pie ───────────────────────────────────────────────────────────
def chart_donut(df, label_col, value_col, title=""):
    tmp = df.groupby(label_col)[value_col].sum().reset_index().nlargest(12, value_col)
    fig = go.Figure(go.Pie(
        labels=tmp[label_col].astype(str).tolist(),
        values=tmp[value_col].tolist(),
        hole=0.52,
        marker=dict(colors=PALETTE[:len(tmp)], line=dict(color=BG0, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT1, size=12),
        hovertemplate="<b>%{label}</b><br>%{value:,.2f} (%{percent})<extra></extra>",
        pull=[0.04 if i == 0 else 0 for i in range(len(tmp))],
    ))
    total = tmp[value_col].sum()
    fig.add_annotation(text=f"{total:,.0f}", x=0.5, y=0.5, font_size=20,
                       font_color=TEXT1, font_family="Space Grotesk, sans-serif",
                       showarrow=False)
    fig.add_annotation(text="total", x=0.5, y=0.4, font_size=11,
                       font_color=TEXT3, showarrow=False)
    _apply(fig, title=title or f"{label_col} composition", showlegend=True)
    _show(fig)


# ── 14. Parallel coordinates ─────────────────────────────────────────────────
def chart_parallel_coords(df, cols: list, color_col=None, title=""):
    tmp = df[cols + ([color_col] if color_col else [])].dropna()
    dims = []
    for c in cols:
        dims.append(dict(label=c, values=tmp[c],
                         range=[tmp[c].min(), tmp[c].max()]))
    color_vals = tmp[color_col] if color_col else tmp[cols[0]]
    fig = go.Figure(go.Parcoords(
        line=dict(color=color_vals, colorscale=SEQUENTIAL,
                  showscale=True, cmin=color_vals.min(), cmax=color_vals.max(),
                  colorbar=dict(title=color_col or cols[0], tickfont=dict(color=TEXT2))),
        dimensions=dims,
        labelfont=dict(color=TEXT2, size=12),
        tickfont =dict(color=TEXT3, size=10),
        rangefont=dict(color=TEXT3, size=9),
    ))
    _apply(fig, title=title or "Parallel coordinates", height=480)
    _show(fig)


# ── 15. Radar / spider ───────────────────────────────────────────────────────
def chart_radar(df, category_col, metric_cols: list, title="", top_n=6):
    cats = df[category_col].value_counts().head(top_n).index.tolist()
    tmp  = df[df[category_col].isin(cats)].groupby(category_col)[metric_cols].mean()
    # Normalize 0-1 per metric
    norm = (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-9)
    fig  = go.Figure()
    for i, cat in enumerate(cats):
        vals = norm.loc[cat].tolist()
        vals += vals[:1]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=metric_cols + metric_cols[:1],
            name=str(cat), fill="toself",
            fillcolor=_rgba(PALETTE[i % len(PALETTE)], 0.2),
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            marker=dict(color=PALETTE[i % len(PALETTE)], size=6),
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>",
        ))
    _apply(fig, title=title or f"Radar: {category_col} vs metrics",
           polar=dict(
               bgcolor=BG2,
               radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                               tickfont=dict(color=TEXT3, size=9)),
               angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT2, size=11)),
           ))
    _show(fig)


# ── 16. Density contour ───────────────────────────────────────────────────────
def chart_density(df, x, y, color_col=None, title=""):
    tmp = df[[x, y]].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=tmp[x], y=tmp[y],
        colorscale=SEQUENTIAL, reversescale=False,
        ncontours=15, contours=dict(showlabels=True, labelfont=dict(color=TEXT1, size=10)),
        hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
        colorbar=dict(title="Density", tickfont=dict(color=TEXT2)),
        line=dict(width=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=tmp[x], y=tmp[y], mode="markers",
        marker=dict(color=ACCENT, size=3, opacity=0.3),
        showlegend=False,
        hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>",
    ))
    _apply(fig, title=title or f"Density: {x} vs {y}")
    _show(fig)


# ── 17. Box plot (styled) ────────────────────────────────────────────────────
def chart_box(df, x, y, title=""):
    cats = df[x].dropna().unique().tolist()[:15]
    fig  = go.Figure()
    for i, cat in enumerate(cats):
        c    = PALETTE[i % len(PALETTE)]
        vals = df[df[x] == cat][y].dropna()
        fig.add_trace(go.Box(
            y=vals, name=str(cat),
            marker=dict(color=c, size=3, opacity=0.7,
                        outliercolor=DANGER, symbol="circle-open"),
            line=dict(color=c, width=1.5),
            fillcolor=_rgba(c, 0.2), boxmean="sd",
            hovertemplate=f"<b>{cat}</b><br>%{{y:.2f}}<extra></extra>",
        ))
    _apply(fig, title=title or f"{y} by {x}", showlegend=False,
           boxmode="group", boxgap=0.2)
    _show(fig)


# ── 18. Candlestick (time series) ────────────────────────────────────────────
def chart_candlestick(df, date_col, open_col, high_col, low_col, close_col, title=""):
    tmp = df[[date_col, open_col, high_col, low_col, close_col]].dropna().sort_values(date_col)
    fig = go.Figure(go.Candlestick(
        x=tmp[date_col], open=tmp[open_col], high=tmp[high_col],
        low=tmp[low_col], close=tmp[close_col],
        increasing=dict(line=dict(color=SUCCESS), fillcolor=_rgba(SUCCESS, 0.5)),
        decreasing=dict(line=dict(color=DANGER),  fillcolor=_rgba(DANGER, 0.5)),
        hovertext=tmp[date_col].astype(str),
    ))
    _apply(fig, title=title or "Candlestick chart",
           xaxis=dict(rangeslider=dict(visible=True, bgcolor=BG2), type="category",
                      gridcolor=BORDER, tickfont=dict(color=TEXT3)))
    _show(fig)


# ── 19. Scatter with regression ──────────────────────────────────────────────
def chart_scatter_regression(df, x, y, color_col=None, title=""):
    tmp = df[[x, y] + ([color_col] if color_col else [])].dropna()
    fig = go.Figure()
    if color_col:
        cats = tmp[color_col].unique()
        for i, cat in enumerate(cats):
            sub = tmp[tmp[color_col] == cat]
            c   = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], name=str(cat), mode="markers",
                marker=dict(color=c, size=6, opacity=0.7, line=dict(color=BG0, width=0.5)),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=tmp[x], y=tmp[y], mode="markers", name="data",
            marker=dict(color=ACCENT, size=6, opacity=0.6, line=dict(color=BG0, width=0.5)),
        ))
    # OLS regression line
    x_arr = tmp[x].values
    y_arr = tmp[y].values
    if np.issubdtype(x_arr.dtype, np.number):
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        r2 = 1 - np.sum((y_arr - np.polyval(coeffs, x_arr))**2) / (np.var(y_arr)*len(y_arr))
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name=f"OLS (R²={r2:.3f})",
            line=dict(color=WARNING, width=2, dash="dash"),
        ))
    _apply(fig, title=title or f"{y} vs {x} with regression")
    _show(fig)


# ── 20. Sankey (flow) ────────────────────────────────────────────────────────
def chart_sankey(df, source_col, target_col, value_col, title=""):
    if source_col == target_col:
        st.warning("Flow analysis requires different Source and Target dimensions.")
        return
    agg = df.groupby([source_col, target_col])[value_col].sum().reset_index()
    labels = list(set(agg[source_col].tolist() + agg[target_col].tolist()))
    label_idx = {l: i for i, l in enumerate(labels)}
    fig = go.Figure(go.Sankey(
        node=dict(
            label=labels,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
            pad=18, thickness=20,
            line=dict(color=BG0, width=0.5),
        ),
        link=dict(
            source=[label_idx[s] for s in agg[source_col]],
            target=[label_idx[t] for t in agg[target_col]],
            value =agg[value_col].tolist(),
            color =[_rgba(PALETTE[i % len(PALETTE)], 0.35) for i in range(len(agg))],
            hovertemplate=f"{source_col}: %{{source.label}}<br>{target_col}: %{{target.label}}<br>Value: %{{value:,.1f}}<extra></extra>",
        ),
        arrangement="snap",
        textfont=dict(color=TEXT1, size=11),
    ))
    _apply(fig, title=title or f"Sankey: {source_col} → {target_col}", height=500)
    _show(fig)


# ── ECharts: Animated gauge ──────────────────────────────────────────────────
def echart_gauge(label: str, value: float, max_val: float):
    if not ECHARTS:
        st.info("Install streamlit-echarts for gauge charts.")
        return
    pct = value / max_val
    color = SUCCESS if pct < 0.5 else WARNING if pct < 0.8 else DANGER
    opts = {
        "backgroundColor": "transparent",
        "series": [{
            "type": "gauge",
            "startAngle": 200, "endAngle": -20,
            "min": 0, "max": round(max_val, 2),
            "splitNumber": 5,
            "progress": {"show": True, "width": 16, "roundCap": True,
                         "itemStyle": {"color": {"type":"linear","x":0,"y":0,"x2":1,"y2":0,
                             "colorStops": [{"offset":0,"color":ACCENT},{"offset":1,"color":color}]}}},
            "pointer": {"show": False},
            "axisLine": {"lineStyle": {"width": 16, "color": [[1, BG2]]}},
            "axisTick": {"show": False}, "splitLine": {"show": False},
            "axisLabel": {"color": TEXT3, "fontSize": 10},
            "title":  {"show": True, "offsetCenter": [0,"28%"],
                       "fontSize": 12, "color": TEXT2},
            "detail": {"valueAnimation": True, "formatter": "{value}",
                       "color": TEXT1, "fontSize": 24, "fontWeight": "bold",
                       "offsetCenter": [0,"5%"]},
            "data": [{"value": round(value, 2), "name": label[:18]}],
        }]
    }
    st_echarts(options=opts, height="200px")


# ── ECharts: Bar race (animated rank) ────────────────────────────────────────
def echart_bar_race(df, category_col, value_col, title=""):
    if not ECHARTS:
        st.info("Install streamlit-echarts for bar race charts.")
        return
    agg = df.groupby(category_col)[value_col].sum().reset_index().sort_values(value_col, ascending=False).head(12)
    labels = agg[category_col].astype(str).tolist()
    values = agg[value_col].round(2).tolist()
    opts = {
        "backgroundColor": "transparent",
        "grid": {"top": 10, "bottom": 30, "left": "22%", "right": "8%"},
        "xAxis": {"max": "dataMax", "axisLabel": {"color": TEXT3, "fontSize": 10},
                  "splitLine": {"lineStyle": {"color": BORDER}}},
        "yAxis": {"type": "category", "data": labels,
                  "inverse": False, "animationDuration": 300,
                  "axisLabel": {"color": TEXT1, "fontSize": 11, "fontWeight": "bold"}},
        "series": [{
            "realtimeSort": True,
            "type": "bar",
            "data": values,
            "label": {"show": True, "position": "right",
                      "formatter": "{c}", "color": TEXT2, "fontSize": 10},
            "itemStyle": {"color": {"type":"linear","x":0,"y":0,"x2":1,"y2":0,
                "colorStops": [{"offset":0,"color":ACCENT},{"offset":1,"color":SUCCESS}]},
                "borderRadius": [0,4,4,0]},
        }],
        "legend": {"show": False},
    }
    st_echarts(options=opts, height="380px")


# ── ECharts: Calendar heatmap ─────────────────────────────────────────────────
def echart_calendar_heatmap(df, date_col, value_col, title=""):
    if not ECHARTS:
        st.info("Install streamlit-echarts for calendar charts.")
        return
    try:
        tmp = df[[date_col, value_col]].dropna().copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        agg = tmp.groupby(tmp[date_col].dt.strftime("%Y-%m-%d"))[value_col].sum().reset_index()
        agg.columns = ["date","val"]
        data = [[row["date"], round(float(row["val"]),2)] for _, row in agg.iterrows()]
        year = agg["date"].str[:4].mode()[0]
        mn, mx = float(agg["val"].min()), float(agg["val"].max())
        opts = {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "item"},
            "visualMap": {
                "min": mn, "max": mx,
                "orient": "horizontal", "left": "center", "bottom": 0,
                "inRange": {"color": [BG2, ACCENT, SUCCESS]},
                "textStyle": {"color": TEXT2},
            },
            "calendar": {
                "range": year,
                "itemStyle": {"borderColor": BG0, "borderWidth": 2},
                "dayLabel": {"color": TEXT3},
                "monthLabel": {"color": TEXT2},
                "yearLabel": {"color": TEXT1, "fontSize": 13},
                "splitLine": {"lineStyle": {"color": BORDER}},
            },
            "series": [{"type":"heatmap","coordinateSystem":"calendar","data":data,
                        "tooltip":{"formatter": "{c[0]}: {c[1]}"}}],
        }
        st_echarts(options=opts, height="200px")
    except Exception as e:
        st.warning(f"Calendar heatmap requires a date column. Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TAB RENDERER — drop-in replacement for the Visualize tab
# ══════════════════════════════════════════════════════════════════════════════
CHART_CATALOG = {
    # Standard
    "Gradient bar":         ("standard",     ["x_cat", "y_num"]),
    "Multi-line":           ("standard",     ["x_any", "y_multi_num"]),
    "Area stack":           ("standard",     ["x_any", "y_multi_num"]),
    "Scatter + regression": ("standard",     ["x_num", "y_num"]),
    "Bubble":               ("standard",     ["x_num", "y_num", "size_num"]),
    # Statistical
    "Histogram + KDE":      ("statistical",  ["x_num"]),
    "Violin + box":         ("statistical",  ["x_cat", "y_num"]),
    "Box plot":             ("statistical",  ["x_cat", "y_num"]),
    "Correlation heatmap":  ("statistical",  []),
    "Scatter matrix":       ("statistical",  ["x_multi_num"]),
    "Density contour":      ("statistical",  ["x_num", "y_num"]),
    # Composition
    "Donut / pie":          ("composition",  ["x_cat", "y_num"]),
    "Treemap":              ("composition",  ["path_cols", "y_num"]),
    "Sunburst":             ("composition",  ["path_cols", "y_num"]),
    "Waterfall":            ("composition",  ["x_cat", "y_num"]),
    "Funnel":               ("composition",  ["x_cat", "y_num"]),
    # Advanced
    "Parallel coordinates": ("advanced",     ["x_multi_num"]),
    "Radar / spider":       ("advanced",     ["x_cat", "y_multi_num"]),
    "Sankey flow":          ("advanced",     ["x_cat", "x_cat2", "y_num"]),
    "Candlestick":          ("advanced",     ["date", "open", "high", "low", "close"]),
    # ECharts
    "Gauge":                ("echarts",      ["y_num"]),
    "Animated bar rank":    ("echarts",      ["x_cat", "y_num"]),
    "Calendar heatmap":     ("echarts",      ["date", "y_num"]),
}

CATEGORIES = ["All", "standard", "statistical", "composition", "advanced", "echarts"]
CAT_LABELS  = {"All":"All", "standard":"Standard", "statistical":"Statistical",
               "composition":"Composition", "advanced":"Advanced", "echarts":"ECharts"}


def render_viz_tab(df: pd.DataFrame, safe_ai_call=None, gemini_key=None, groq_key=None):
    """
    Main entry point. Call from your app.py Visualize tab:
        from visualization import render_viz_tab
        with tab_viz:
            render_viz_tab(df, safe_ai_call=safe_ai_call)
    """
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object","category"]).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "day" in c.lower()]
    all_cols  = df.columns.tolist()

    # ── Category filter ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:11px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{TEXT3};margin-bottom:10px">Chart category</div>',
        unsafe_allow_html=True,
    )
    cat_filter = st.radio(
        "Filter",
        options=CATEGORIES,
        format_func=lambda x: CAT_LABELS[x],
        horizontal=True,
        label_visibility="collapsed",
    )

    visible = {k: v for k, v in CHART_CATALOG.items()
               if cat_filter == "All" or v[0] == cat_filter}

    # ── Chart picker ─────────────────────────────────────────────────────────
    st.markdown(f'<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    chart_name = st.selectbox(
        "Chart type",
        list(visible.keys()),
        label_visibility="collapsed",
    )

    _, needs = CHART_CATALOG[chart_name]
    cat_str = CHART_CATALOG[chart_name][0]
    cat_colors = {
        "standard":"chip-accent","statistical":"chip-success",
        "composition":"chip-warning","advanced":"chip-danger","echarts":"chip-neutral",
    }
    st.markdown(
        f'<span class="chip {cat_colors.get(cat_str,"chip-neutral")}">{cat_str}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div style="margin-bottom:12px"></div>', unsafe_allow_html=True)

    # ── Sampling ─────────────────────────────────────────────────────────────
    max_pts = st.slider("Max data points", 500, 100_000, 20_000, step=500,
                        help="Sampling prevents browser slowdown on large datasets")
    plot_df = df.sample(n=min(len(df), max_pts), random_state=42) if len(df) > max_pts else df
    if len(df) > max_pts:
        st.caption(f"Showing {max_pts:,} of {len(df):,} rows (random sample)")

    st.divider()

    # ── Per-chart controls & render ──────────────────────────────────────────
    try:

        # ── GRADIENT BAR ──
        if chart_name == "Gradient bar":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category axis", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_bar_gradient(plot_df, x, y)

        # ── MULTI-LINE ──
        elif chart_name == "Multi-line":
            c1, c2 = st.columns([2,3])
            x = c1.selectbox("X axis", all_cols)
            ys = c2.multiselect("Y columns (select 1–5)", num_cols, default=num_cols[:2])
            if ys and st.button("Plot ↗", key="plot_main"):
                chart_multiline(plot_df, x, ys[:5])

        # ── AREA STACK ──
        elif chart_name == "Area stack":
            c1, c2 = st.columns([2,3])
            x  = c1.selectbox("X axis", all_cols)
            ys = c2.multiselect("Y columns", num_cols, default=num_cols[:3])
            if ys and st.button("Plot ↗", key="plot_main"):
                chart_area_stack(plot_df, x, ys)

        # ── SCATTER + REGRESSION ──
        elif chart_name == "Scatter + regression":
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X (numeric)", num_cols)
            y = c2.selectbox("Y (numeric)", num_cols, index=min(1, len(num_cols)-1))
            col = c3.selectbox("Color by", ["None"] + cat_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_scatter_regression(plot_df, x, y, col if col != "None" else None)

        # ── BUBBLE ──
        elif chart_name == "Bubble":
            c1, c2, c3, c4 = st.columns(4)
            x    = c1.selectbox("X", num_cols)
            y    = c2.selectbox("Y", num_cols, index=min(1,len(num_cols)-1))
            size = c3.selectbox("Size", num_cols, index=min(2,len(num_cols)-1))
            col  = c4.selectbox("Color by", ["None"] + cat_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_bubble(plot_df, x, y, size, col if col != "None" else None)

        # ── HISTOGRAM + KDE ──
        elif chart_name == "Histogram + KDE":
            c1, c2 = st.columns(2)
            col   = c1.selectbox("Column", num_cols)
            nbins = c2.slider("Bins", 10, 100, 35)
            if st.button("Plot ↗", key="plot_main"):
                chart_histogram_kde(plot_df, col, nbins)

        # ── VIOLIN + BOX ──
        elif chart_name == "Violin + box":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_violin_box(plot_df, x, y)

        # ── BOX PLOT ──
        elif chart_name == "Box plot":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_box(plot_df, x, y)

        # ── CORRELATION HEATMAP ──
        elif chart_name == "Correlation heatmap":
            if st.button("Plot ↗", key="plot_main"):
                chart_correlation(plot_df)

        # ── SCATTER MATRIX ──
        elif chart_name == "Scatter matrix":
            c1, c2 = st.columns(2)
            cols_sel = c1.multiselect("Columns (3–5)", num_cols, default=num_cols[:4])
            col_c    = c2.selectbox("Color by", ["None"] + cat_cols)
            if len(cols_sel) >= 2 and st.button("Plot ↗", key="plot_main"):
                chart_splom(plot_df, cols_sel[:5], col_c if col_c != "None" else None)

        # ── DENSITY CONTOUR ──
        elif chart_name == "Density contour":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X", num_cols)
            y = c2.selectbox("Y", num_cols, index=min(1,len(num_cols)-1))
            if st.button("Plot ↗", key="plot_main"):
                chart_density(plot_df, x, y)

        # ── DONUT ──
        elif chart_name == "Donut / pie":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Labels", cat_cols or all_cols)
            y = c2.selectbox("Values", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_donut(plot_df, x, y)

        # ── TREEMAP ──
        elif chart_name == "Treemap":
            c1, c2 = st.columns(2)
            paths = c1.multiselect("Hierarchy (2 levels)", cat_cols, default=cat_cols[:2])
            val   = c2.selectbox("Value", num_cols)
            if len(paths) >= 1 and st.button("Plot ↗", key="plot_main"):
                chart_treemap(plot_df, paths, val)

        # ── SUNBURST ──
        elif chart_name == "Sunburst":
            c1, c2 = st.columns(2)
            paths = c1.multiselect("Hierarchy (2 levels)", cat_cols, default=cat_cols[:2])
            val   = c2.selectbox("Value", num_cols)
            if len(paths) >= 1 and st.button("Plot ↗", key="plot_main"):
                chart_sunburst(plot_df, paths, val)

        # ── WATERFALL ──
        elif chart_name == "Waterfall":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_waterfall(plot_df, x, y)

        # ── FUNNEL ──
        elif chart_name == "Funnel":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Stage", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_funnel(plot_df, x, y)

        # ── PARALLEL COORDS ──
        elif chart_name == "Parallel coordinates":
            c1, c2 = st.columns(2)
            cols_p = c1.multiselect("Numeric dimensions (3+)", num_cols, default=num_cols[:5])
            col_c  = c2.selectbox("Color by", ["None"] + num_cols)
            if len(cols_p) >= 2 and st.button("Plot ↗", key="plot_main"):
                chart_parallel_coords(plot_df, cols_p, col_c if col_c != "None" else None)

        # ── RADAR ──
        elif chart_name == "Radar / spider":
            c1, c2, c3 = st.columns(3)
            cat   = c1.selectbox("Category", cat_cols or all_cols)
            mets  = c2.multiselect("Metrics (3–6)", num_cols, default=num_cols[:4])
            top_n = c3.slider("Top N categories", 2, 8, 5)
            if len(mets) >= 3 and st.button("Plot ↗", key="plot_main"):
                chart_radar(plot_df, cat, mets, top_n=top_n)

        # ── SANKEY ──
        elif chart_name == "Sankey flow":
            c1, c2, c3 = st.columns(3)
            src = c1.selectbox("Source", cat_cols or all_cols)
            tgt = c2.selectbox("Target", [c for c in (cat_cols or all_cols) if c != src] or all_cols)
            val = c3.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_sankey(plot_df, src, tgt, val)

        # ── CANDLESTICK ──
        elif chart_name == "Candlestick":
            st.markdown(f'<div style="font-size:12px;color:{TEXT2};margin-bottom:8px">Requires date, open, high, low, close columns.</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            d_col = c1.selectbox("Date",  date_cols or all_cols)
            o_col = c2.selectbox("Open",  num_cols)
            h_col = c3.selectbox("High",  num_cols)
            l_col = c4.selectbox("Low",   num_cols)
            cl_col= c5.selectbox("Close", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                chart_candlestick(plot_df, d_col, o_col, h_col, l_col, cl_col)

        # ── GAUGE ──
        elif chart_name == "Gauge":
            c1, c2 = st.columns(2)
            gcols = c1.multiselect("Metrics to gauge (1–4)", num_cols, default=num_cols[:3])
            if gcols and st.button("Plot ↗", key="plot_main"):
                cols_ = st.columns(len(gcols))
                for i, gc in enumerate(gcols):
                    with cols_[i]:
                        echart_gauge(gc, float(df[gc].mean()), float(df[gc].max()))

        # ── BAR RACE ──
        elif chart_name == "Animated bar rank":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category", cat_cols or all_cols)
            y = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                echart_bar_race(plot_df, x, y)

        # ── CALENDAR HEATMAP ──
        elif chart_name == "Calendar heatmap":
            c1, c2 = st.columns(2)
            d = c1.selectbox("Date column", date_cols or all_cols)
            v = c2.selectbox("Value", num_cols)
            if st.button("Plot ↗", key="plot_main"):
                echart_calendar_heatmap(plot_df, d, v)

    except Exception as e:
        st.error(f"Chart error: {e}")

    st.divider()

    # ── AI recommendations ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:11px;font-weight:500;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{TEXT3};margin-bottom:8px">AI chart recommendations</div>',
        unsafe_allow_html=True,
    )
    ai_available = safe_ai_call is not None and (gemini_key or groq_key)
    if not ai_available:
        st.markdown(f'<div style="font-size:13px;color:{TEXT3}">Add Gemini or Groq API key to enable AI recommendations.</div>', unsafe_allow_html=True)
    else:
        if st.button("Generate AI recommendations ↗", key="ai_rec"):
            schema = df.dtypes.to_string()
            with st.spinner("AI analysing your dataset…"):
                try:
                    raw, used_model = safe_ai_call(
                        f"Dataset schema:\n{schema}\n\n"
                        "Recommend 3 insightful charts. Return ONLY a JSON array. "
                        "Each object must have: title, chart_type, x, y (null if not needed), rationale. "
                        "chart_type options: scatter, bar, histogram, box, violin, treemap, sunburst, line, area, density, radar, waterfall, bubble. "
                        "Focus on the most analytically valuable insights, not just the simplest charts."
                    )
                    clean = raw.strip().replace("```json","").replace("```","")
                    recs  = json.loads(clean)
                    st.session_state["viz_recs"] = recs
                    st.session_state["viz_rec_model"] = used_model
                except Exception as e:
                    st.error(f"Could not parse AI recommendations: {e}")

        if st.session_state.get("viz_recs"):
            model_str = st.session_state.get("viz_rec_model","")
            st.markdown(
                f'<div class="model-badge">▲ powered by <span class="model-name">'
                f'{model_str.replace("models/","").replace("groq/","⚡ ")}</span></div>',
                unsafe_allow_html=True,
            )
            for i, rec in enumerate(st.session_state["viz_recs"]):
                with st.expander(f"◈ Recommendation {i+1}: {rec['title']}", expanded=(i==0)):
                    st.markdown(
                        f'<div style="font-size:13px;color:{TEXT2};margin-bottom:10px">{rec["rationale"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Render chart", key=f"render_rec_{i}"):
                        st.session_state[f"show_rec_{i}"] = True
                    if st.session_state.get(f"show_rec_{i}"):
                        rx, ry, rt = rec.get("x"), rec.get("y"), rec.get("chart_type","bar")
                        # Helper to resolve columns or calculations
                        def res_col(target: str, df_internal: pd.DataFrame):
                            if not target: return None
                            if target in df_internal.columns: return target
                            # Try simple ratios: "a / b"
                            if " / " in target:
                                parts = target.split(" / ")
                                if all(p in df_internal.columns for p in parts):
                                    new_col = f"calc_{target.replace(' ','_')}"
                                    df_internal[new_col] = df_internal[parts[0]] / (df_internal[parts[1]] + 1e-9)
                                    return new_col
                            return None

                        try:
                            df_work = df.copy()
                            # Enhanced resolution (handles strings and lists)
                            def resolve_ai_target(t, df_):
                                if isinstance(t, list):
                                    return [res_col(i, df_) or i for i in t]
                                return res_col(t, df_) or t

                            rx_res = resolve_ai_target(rx, df_work)
                            ry_res = resolve_ai_target(ry, df_work)
                            
                            if rt == "bar"         and rx_res and ry_res: chart_bar_gradient(df_work, rx_res, ry_res)
                            elif rt == "scatter"   and rx_res and ry_res: chart_scatter_regression(df_work, rx_res, ry_res)
                            elif rt == "histogram" and rx_res:            chart_histogram_kde(df_work, rx_res)
                            elif rt == "box"       and rx_res and ry_res: chart_box(df_work, rx_res, ry_res)
                            elif rt == "violin"    and rx_res and ry_res: chart_violin_box(df_work, rx_res, ry_res)
                            elif rt == "treemap"   and rx_res and ry_res: 
                                p_ = rx_res if isinstance(rx_res, list) else [rx_res]
                                chart_treemap(df_work, p_, ry_res)
                            elif rt == "sunburst"  and rx_res and ry_res: 
                                p_ = rx_res if isinstance(rx_res, list) else [rx_res]
                                chart_sunburst(df_work, p_, ry_res)
                            elif rt == "line"      and rx_res and ry_res: chart_multiline(df_work, rx_res, [ry_res])
                            elif rt == "area"      and rx_res and ry_res: chart_area_stack(df_work, rx_res, [ry_res])
                            elif rt == "density"   and rx_res and ry_res: chart_density(df_work, rx_res, ry_res)
                            elif rt == "waterfall" and rx_res and ry_res: chart_waterfall(df_work, rx_res, ry_res)
                            elif rt == "bubble"    and rx_res and ry_res: chart_bubble(df_work, rx_res, ry_res, ry_res)
                            else:
                                fig = px.bar(df, x=rx, y=ry, title=rec["title"], color_discrete_sequence=PALETTE) if ry else px.histogram(df, x=rx, title=rec["title"], color_discrete_sequence=PALETTE)
                                _apply(fig)
                                _show(fig)
                        except Exception as e:
                            st.error(f"Render error: {e}")
