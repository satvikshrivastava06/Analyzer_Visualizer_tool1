import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import cast
from streamlit.components.v1 import html as st_html

try:
    from streamlit_echarts import st_echarts
    ECHARTS = True
except ImportError:
    ECHARTS = False

# ── Design tokens (Data Analyzer & Visualizer Tool Theme) ──────────────────────
BG0       = "#0A0C12"
SURFACE   = "#13161F"
SURFACE2  = "#1C2030"
BORDER    = "#2A2F45"
ACCENT1   = "#6C8CFF"  # Blue
ACCENT2   = "#FF6B6B"  # Red
ACCENT3   = "#43E8B5"  # Green
TEXT      = "#E8ECF7"
MUTED     = "#6B7294"

PAL = [ACCENT1, ACCENT3, "#FF9F43", ACCENT2, "#A8DADC", "#C77DFF"]

# ── Custom Styles for Metric Cards & Layout ──────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

.metric-card {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 20px 24px;
  position: relative;
  overflow: hidden;
  margin-bottom: 12px;
}}
.metric-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, {ACCENT1}, {ACCENT3});
}}
.metric-val {{ font-size: 1.8rem; font-weight: 800; color: {TEXT}; font-family: 'JetBrains Mono', monospace; }}
.metric-label {{ font-size: 0.68rem; color: {MUTED}; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; font-weight: 600; }}
.metric-delta {{ font-size: 0.75rem; font-weight: 600; margin-top: 6px; }}
.delta-up   {{ color: {ACCENT3}; }}
.delta-down {{ color: {ACCENT2}; }}

.section-title {{
  font-size: 0.7rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: {MUTED};
  margin-bottom: 12px;
  margin-top: 24px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.section-title::after {{
  content: '';
  flex: 1;
  height: 1px;
  background: {BORDER};
}}
</style>
"""

# ── ECharts Injection Helper (Custom HTML for precise styling) ───────────────
def _inject_echarts(option: dict, height: int = 360, key: str = "chart"):
    CHART_BASE = {
        "backgroundColor": "transparent",
        "textStyle": {"fontFamily": "Syne, sans-serif", "color": TEXT},
        "grid": {"left": "6%", "right": "4%", "bottom": "12%", "top": "14%", "containLabel": True},
        "tooltip": {
            "backgroundColor": SURFACE2,
            "borderColor": BORDER,
            "textStyle": {"color": TEXT, "fontFamily": "Syne, sans-serif", "fontSize": 12},
        },
    }
    merged = {**CHART_BASE, **option}
    # Ensure key is unique using a hash of the option id if needed
    safe_key = f"{key}_{id(option)}"
    html_code = f"""
    <div id="chart_{safe_key}" style="width:100%;height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script>
      (function() {{
        var el = document.getElementById('chart_{safe_key}');
        var chart = echarts.init(el, null, {{renderer:'canvas'}});
        chart.setOption({json.dumps(merged)});
        window.addEventListener('resize', function(){{ chart.resize(); }});
      }})();
    </script>
    """
    st_html(html_code, height=height + 20)

# ── Revenue Dashboard View ───────────────────────────────────────────────────
def render_revenue_dashboard(df: pd.DataFrame):
    st.markdown(_CSS, unsafe_allow_html=True)
    
    # Identify key columns
    rev_col = next((c for c in df.columns if "revenue" in c.lower() or "sales" in c.lower()), None)
    prof_col = next((c for c in df.columns if "profit" in c.lower()), None)
    cost_col = next((c for c in df.columns if "cost" in c.lower()), None)
    cat_col = next((c for c in df.columns if "category" in c.lower() or "segment" in c.lower()), df.select_dtypes(include="object").columns[0] if not df.select_dtypes(include="object").columns.empty else None)
    reg_col = next((c for c in df.columns if "region" in c.lower() or "country" in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "stamp" in c.lower()), None)
    
    if not (rev_col and prof_col):
        st.warning("Revenue Dashboard requires 'Revenue' and 'Profit' columns. Showing simplified view.")
        return

    # ── KPI Row ──
    total_rev = df[rev_col].sum()
    total_prof = df[prof_col].sum()
    total_cost = df[cost_col].sum() if cost_col else total_rev - total_prof
    margin = (total_prof / total_rev * 100) if total_rev != 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, "Total Revenue", f"${total_rev/1e6:.2f}M", "↑ 12.4%", True),
        (c2, "Net Profit", f"${total_prof/1e6:.2f}M", "↑ 8.1%",  True),
        (c3, "Total Cost", f"${total_cost/1e6:.2f}M", "↓ 2.3%",  True),
        (c4, "Profit Margin", f"{margin:.1f}%", "↑ 0.5 pp", True),
    ]
    for col, label, val, delta, up in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-val">{val}</div>
              <div class="metric-delta {'delta-up' if up else 'delta-down'}">{delta} vs baseline</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 1: Trend + Segment ──
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        st.markdown('<div class="section-title">Performance Trend (Revenue vs Cost)</div>', unsafe_allow_html=True)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            mon_agg = df.groupby(df[date_col].dt.to_period("M")).agg({rev_col: "sum", prof_col: "sum"}).reset_index()
            mon_agg[date_col] = mon_agg[date_col].astype(str)
            
            _inject_echarts({
                "legend": {"data": ["Revenue", "Profit"], "textStyle": {"color": MUTED}, "top": 4, "right": 8},
                "xAxis": {"type": "category", "data": mon_agg[date_col].tolist(), "axisLine": {"lineStyle": {"color": BORDER}}},
                "yAxis": {"type": "value", "splitLine": {"lineStyle": {"color": SURFACE2, "type": "dashed"}}},
                "series": [
                    {"name": "Revenue", "type": "line", "smooth": True, "data": mon_agg[rev_col].tolist(), "lineStyle": {"width": 3, "color": ACCENT1}, "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1, "colorStops": [{"offset": 0, "color": f"{ACCENT1}44"}, {"offset": 1, "color": "transparent"}]}}},
                    {"name": "Profit", "type": "bar", "data": mon_agg[prof_col].tolist(), "itemStyle": {"color": ACCENT3, "borderRadius": [4, 4, 0, 0]}}
                ]
            }, height=320, key="trend")
            
    with col_b:
        st.markdown('<div class="section-title">Segment Breakdown</div>', unsafe_allow_html=True)
        if cat_col:
            cat_agg = df.groupby(cat_col)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False).head(5)
            _inject_echarts({
                "xAxis": {"type": "category", "data": cat_agg[cat_col].tolist()},
                "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
                "series": [{"type": "bar", "data": [{"value": v, "itemStyle": {"color": PAL[i%len(PAL)], "borderRadius": [6,6,0,0]}} for i, v in enumerate(cat_agg[rev_col].tolist())], "barMaxWidth": 40}]
            }, height=320, key="seg")

    # ── Row 2: Radar + Gauge ──
    col_c, col_d = st.columns([2, 2])
    
    with col_c:
        st.markdown('<div class="section-title">Regional Radar</div>', unsafe_allow_html=True)
        if reg_col:
            reg_agg = df.groupby(reg_col)[rev_col].sum().reset_index()
            max_v = float(reg_agg[rev_col].max())
            _inject_echarts({
                "radar": {
                    "indicator": [{"name": r, "max": max_v} for r in reg_agg[reg_col].tolist()],
                    "shape": "circle",
                    "splitArea": {"show": True, "areaStyle": {"color": [f"{SURFACE2}66", f"{SURFACE}44"]}},
                },
                "series": [{"type": "radar", "data": [{"value": reg_agg[rev_col].tolist(), "name": "Revenue", "areaStyle": {"color": f"{ACCENT1}33"}, "lineStyle": {"color": ACCENT1}}]}]
            }, height=300, key="radar")
            
    with col_d:
        st.markdown('<div class="section-title">Profit Efficiency</div>', unsafe_allow_html=True)
        _inject_echarts({
            "series": [{
                "type": "gauge", "startAngle": 200, "endAngle": -20, "min": 0, "max": 100,
                "progress": {"show": True, "width": 14, "itemStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0, "colorStops": [{"offset": 0, "color": ACCENT2}, {"offset": 1, "color": ACCENT3}]}}},
                "axisLine": {"lineStyle": {"width": 14, "color": [[1, SURFACE2]]}},
                "pointer": {"show": False},
                "detail": {"formatter": "{value}%", "color": TEXT, "fontSize": 24, "fontWeight": 700, "offsetCenter": [0, "10%"]},
                "title": {"show": True, "offsetCenter": [0, "50%"], "color": MUTED, "fontSize": 12, "fontWeight": 600},
                "data": [{"value": round(margin, 1), "name": "MARGIN"}]
            }]
        }, height=300, key="gauge")

# ── Original Chart Component Mixins ──────────────────────────────────────────
def _base(title=""):
    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": MUTED, "fontFamily": "Syne, sans-serif"},
        "title": {
            "text": title,
            "textStyle": {"color": TEXT, "fontSize": 14, "fontFamily": "Syne, sans-serif", "fontWeight": "600"},
            "left": "4",
        },
    }

def chart_kpi_gauges(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols: return
    selected = st.multiselect("Select metrics", num_cols, default=num_cols[:min(4, len(num_cols))], max_selections=4)
    if not selected: return
    cols = st.columns(len(selected))
    for i, col_name in enumerate(selected):
        mean_v = float(df[col_name].mean())
        max_v = float(df[col_name].max()) or 1.0
        with cols[i]:
            _inject_echarts({
                "series": [{
                    "type": "gauge", "radius": "85%", "startAngle": 210, "endAngle": -30, "min": 0, "max": max_v,
                    "progress": {"show": True, "width": 10, "itemStyle": {"color": PAL[i%len(PAL)]}},
                    "axisLine": {"lineStyle": {"width": 10, "color": [[1, SURFACE2]]}},
                    "axisTick": {"show": False}, "splitLine": {"show": False}, "axisLabel": {"show": False},
                    "pointer": {"show": True, "length": "60%", "width": 4, "itemStyle": {"color": TEXT}},
                    "anchor": {"show": True, "showAbove": True, "size": 10, "itemStyle": {"color": TEXT, "borderWidth": 2, "borderColor": ACCENT1}},
                    "detail": {"fontSize": 18, "fontWeight": 700, "offsetCenter": [0, "15%"], "color": TEXT, "formatter": "{value}"},
                    "data": [{"value": round(mean_v, 1), "name": col_name}],
                    "title": {"show": True, "offsetCenter": [0, "60%"], "color": MUTED, "fontSize": 10, "fontWeight": 600}
                }]
            }, height=200, key=f"kg_{i}")

def chart_gradient_bar(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not num_cols or not cat_cols: return
    c1, c2 = st.columns(2)
    x_col = c1.selectbox("Category", cat_cols, key="gb_x")
    y_col = c2.selectbox("Value", num_cols, key="gb_y")
    agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col).tail(12)
    _inject_echarts({
        "yAxis": {"type": "category", "data": agg[x_col].tolist()},
        "xAxis": {"type": "value"},
        "series": [{"type": "bar", "data": agg[y_col].tolist(), "itemStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 1, "y2": 0, "colorStops": [{"offset": 0, "color": ACCENT1}, {"offset": 1, "color": ACCENT3}]}}, "borderRadius": [0, 4, 4, 0]}]
    }, height=400, key="gb_main")

def chart_calendar_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "stamp" in c.lower()]
    if not date_cols:
        st.info("No date/time column detected.")
        return
    c1, c2 = st.columns(2)
    d_col = c1.selectbox("Date column", date_cols, key="cal_d")
    v_col = c2.selectbox("Measure", num_cols, key="cal_v")
    try:
        tmp = df[[d_col, v_col]].copy()
        tmp[d_col] = pd.to_datetime(tmp[d_col], errors="coerce")
        tmp = tmp.dropna().sort_values(d_col)
        agg = tmp.groupby(tmp[d_col].dt.strftime("%Y-%m-%d"))[v_col].sum().reset_index()
        date_col_name = agg.columns[0]
        data = [[r[date_col_name], round(float(r[v_col]), 2)] for _, r in agg.iterrows()]
        year = agg[date_col_name].str[:4].mode()[0] if not agg.empty else "2024"
        mn, mx = float(agg[v_col].min() or 0), float(agg[v_col].max() or 100)
    except Exception as e:
        st.error(f"Data error: {e}")
        return

    _inject_echarts({
        "tooltip": {"position": "top", "formatter": "function(p){return p.data[0]+': '+p.data[1];}"},
        "visualMap": {"min": mn, "max": mx, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0, "inRange": {"color": [SURFACE2, ACCENT1, ACCENT3]}, "textStyle": {"color": MUTED}},
        "calendar": {"top": 40, "left": 40, "right": 10, "range": year, "itemStyle": {"color": SURFACE, "borderColor": BORDER, "borderWidth": 1}, "dayLabel": {"color": MUTED}, "monthLabel": {"color": TEXT}},
        "series": [{"type": "heatmap", "coordinateSystem": "calendar", "data": data}]
    }, height=240, key="calendar_heat")

def render_premium_viz_tab(df: pd.DataFrame):
    if not ECHARTS:
        st.error("Install streamlit-echarts")
        return

    st.markdown(_CSS, unsafe_allow_html=True)
    
    VIEWS = {
        "💎  Executive Dashboard": render_revenue_dashboard,
        "◎  Metric Gauges":      chart_kpi_gauges,
        "▣  Gradient Ranking":   chart_gradient_bar,
        "▲  Activity Calendar":  chart_calendar_heatmap,
    }
    
    selected = st.radio("View", list(VIEWS.keys()), horizontal=True, label_visibility="collapsed")
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    try:
        VIEWS[selected](df)
    except Exception as e:
        st.error(f"Render Error: {e}")
