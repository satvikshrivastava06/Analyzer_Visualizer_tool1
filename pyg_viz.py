import streamlit as st
import pandas as pd
import json
import pygwalker as pyg
from streamlit.components.v1 import html as st_html

# ── Design tokens (Data Analyzer & Visualizer Tool Theme) ──────────────────────
BG0       = "#0A0C12"
SURFACE   = "#13161F"
BORDER    = "#2A2F45"
ACCENT1   = "#6C8CFF"
MUTED     = "#6B7294"

# Known longitude/latitude column name patterns
LAT_PATTERNS  = ["lat", "latitude", "Latitude", "LAT", "y", "Y"]
LON_PATTERNS  = ["lon", "lng", "longitude", "Longitude", "LON", "LNG", "x", "X"]

def _detect_geo_columns(df: pd.DataFrame):
    """Return (lat_col, lon_col) if plausible geo columns exist, else (None, None)."""
    lat_col = next((c for c in df.columns if c.lower() in [p.lower() for p in LAT_PATTERNS]), None)
    lon_col = next((c for c in df.columns if c.lower() in [p.lower() for p in LON_PATTERNS]), None)
    return lat_col, lon_col


def render_walker_tab(df: pd.DataFrame):
    """
    Data Analyzer & Visualizer Tool PyGWalker Integration.
    Drop-in replacement for BI Explorer tab.
    """
    st.markdown(f"""
    <div style="margin-bottom:16px;">
      <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; color:{MUTED}; font-weight:700; display:flex; align-items:center; gap:8px;">
        🔭  Drag-and-Drop Data Explorer
        <div style="flex:1; height:1px; background:{BORDER};"></div>
      </div>
      <p style="font-size:0.82rem; color:{MUTED}; margin:4px 0 0 0;">
        Drag fields to axes, color, and size channels. Use the toolbar to toggle between chart types and table views.
        Enable <strong style="color:{ACCENT1}">Geo Map Mode</strong> to plot data on a world map.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Launch Flag
    if 'pyg_launched' not in st.session_state:
        st.session_state.pyg_launched = False
    if 'pyg_geo_mode' not in st.session_state:
        st.session_state.pyg_geo_mode = False

    if not st.session_state.pyg_launched:
        st.markdown(f"""
            <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:32px; text-align:center; margin-bottom:20px">
                <div style="font-size:32px; margin-bottom:16px">🔭</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:600; color:#F0F2FF; margin-bottom:8px">
                    Initialize BI Explorer
                </div>
                <div style="font-size:13px; color:#8B90B4; margin-bottom:20px">
                    Launch the interactive drag-and-drop exploration engine (PyGWalker).<br>
                    Includes world map support for datasets with latitude/longitude columns.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Explorer ↗", key="launch_pyg_actual"):
            st.session_state.pyg_launched = True
            st.rerun()
    else:
        # ── Geo Mode Toggle ──────────────────────────────────────────────────
        lat_col, lon_col = _detect_geo_columns(df)
        has_geo = lat_col is not None and lon_col is not None

        col_toggle, col_info = st.columns([1, 3])
        with col_toggle:
            geo_mode = st.toggle(
                "🌍 Geo Map Mode",
                value=st.session_state.pyg_geo_mode,
                key="pyg_geo_toggle",
                help="Switch the coordinate system to a world map projection. Requires latitude & longitude columns.",
                disabled=not has_geo,
            )
            if geo_mode != st.session_state.pyg_geo_mode:
                st.session_state.pyg_geo_mode = geo_mode
                st.rerun()

        with col_info:
            if has_geo:
                st.markdown(
                    f'<div style="font-size:12px; color:#8B90B4; margin-top:8px;">'
                    f'✅ Detected geo columns: <code style="color:{ACCENT1}">{lat_col}</code> (lat) · '
                    f'<code style="color:{ACCENT1}">{lon_col}</code> (lon)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:12px; color:#4B5073; margin-top:8px;">'
                    '⚠️ No lat/lon columns detected. Geo Map Mode is disabled.</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='margin:12px 0; border-color:#2A2F45'>", unsafe_allow_html=True)

        try:
            # ── Build spec ──────────────────────────────────────────────────
            if st.session_state.pyg_geo_mode and has_geo:
                # Geographic coordinate system spec for world map plotting
                spec_dict = {
                    "config": {
                        "defaultAggregated": False,
                        "geoms": ["poi"],          # Point of Interest — dots on a map
                        "coordSystem": "geographic",
                    },
                    "encodings": {
                        "longitude": [{"field": lon_col, "type": "quantitative"}],
                        "latitude":  [{"field": lat_col,  "type": "quantitative"}],
                        "color":     [],
                        "size":      [],
                    },
                }
            else:
                # Standard Cartesian spec
                spec_dict = {
                    "config": {
                        "defaultAggregated": True,
                        "geoms": ["auto"],
                        "coordSystem": "generic",
                    },
                    "encodings": {
                        "color": [{"field": df.columns[0], "type": "nominal"}] if not df.empty else []
                    },
                }

            walker_html = pyg.to_html(
                df,
                spec=json.dumps(spec_dict),
                theme_key="dark",
                dark="dark",
                appearance="dark",
            )

            # ── Inject with styling ──────────────────────────────────────
            st_html(
                f"""
                <style>
                  body {{ background: transparent; overflow: hidden; }}
                  .gw-app {{ background: {SURFACE} !important; border-radius: 12px; overflow: hidden; border: 1px solid {BORDER}; }}
                  .gw-toolbar {{ background: #1C2030 !important; }}
                  /* Map tile layer dark mode */
                  .leaflet-tile {{ filter: brightness(0.85) saturate(0.8) !important; }}
                </style>
                {walker_html}
                """,
                height=800,
                scrolling=False,
            )
        except Exception as e:
            st.error(f"Explorer Error: {e}")
            if st.button("Reset Session"):
                st.session_state.pyg_launched = False
                st.session_state.pyg_geo_mode = False
                st.rerun()
