"""
Page 1: Interactive Coverage Map
================================
Folium map showing all schools with MSHP status and health burden overlay.
"""
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Coverage Map | MSHP Analysis",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { 
        font-family: 'Inter', system-ui, sans-serif; 
        background-color: #ffffff;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ensure readable text in light theme */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label { 
        color: #333333 !important; 
    }
    .stApp h1, .stApp h2, .stApp h3 { 
        color: #1a1a2e !important; 
    }
    
    [data-testid="stSidebar"] { 
        background-color: #f8f9fa !important; 
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] * { color: #333333 !important; }
    [data-testid="stSidebar"] a { color: #0054a3 !important; }
    
    /* Make captions readable */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stCaptionContainer"] span,
    .stApp .stCaption,
    .stApp small,
    [data-testid="stCaptionContainer"] *,
    .stCaption p,
    div[data-testid="stCaptionContainer"] p,
    .st-emotion-cache-nahz7x,
    .st-emotion-cache-eczf16,
    .st-emotion-cache-1gulkj5,
    .st-emotion-cache-6q9sum,
    .st-emotion-cache-1wivap2 { color: #666666 !important; }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, load_uhf_boundaries, get_neighborhood_summary
from utils.charts import COLORS

df = load_schools_data()
neighborhood_summary = get_neighborhood_summary(df)

st.title("Interactive Coverage Map")
st.markdown("Explore MSHP coverage across Bronx schools with health burden overlay.")

st.sidebar.header("Map Filters")

show_mshp = st.sidebar.checkbox("Show MSHP Schools", value=True)
show_non_mshp = st.sidebar.checkbox("Show Non-MSHP Schools", value=True)

tier_filter = st.sidebar.multiselect(
    "Priority Tier (Non-MSHP only)",
    options=['Tier 1 - Critical', 'Tier 2 - High', 'Tier 3 - Moderate', 'Tier 4 - Lower'],
    default=['Tier 1 - Critical', 'Tier 2 - High', 'Tier 3 - Moderate', 'Tier 4 - Lower']
)

show_choropleth = st.sidebar.checkbox("Show Asthma Burden Layer", value=True)
use_clustering = st.sidebar.checkbox("Cluster Markers", value=False)

show_deserts = st.sidebar.checkbox("Highlight Health Deserts", value=False)
desert_threshold = 1.0
if show_deserts:
    desert_threshold = st.sidebar.slider(
        "Desert threshold (miles to nearest MSHP)",
        min_value=0.25, max_value=2.0, value=1.0, step=0.25,
    )

filtered_df = df.copy()
if not show_mshp:
    filtered_df = filtered_df[~filtered_df['has_mshp']]
if not show_non_mshp:
    filtered_df = filtered_df[filtered_df['has_mshp']]
if 'priority_tier' in filtered_df.columns and tier_filter:
    filtered_df = filtered_df[
        (filtered_df['has_mshp']) | (filtered_df['priority_tier'].isin(tier_filter))
    ]

bronx_center = [40.85, -73.85]
m = folium.Map(location=bronx_center, zoom_start=12, tiles='CartoDB dark_matter')

if show_choropleth:
    try:
        gdf = load_uhf_boundaries()
        gdf = gdf.merge(
            neighborhood_summary[['uhf_name', 'asthma_rate', 'coverage_pct']],
            on='uhf_name', how='left'
        )
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=['uhf_name', 'asthma_rate'],
            key_on='feature.properties.uhf_name',
            fill_color='YlOrRd',
            fill_opacity=0.4,
            line_opacity=0.6,
            legend_name='Asthma ED Visits (per 10K)',
            name='Asthma Burden'
        ).add_to(m)
    except Exception as e:
        st.sidebar.warning(f"Could not load choropleth: {e}")

def get_marker_color(row):
    if row['has_mshp']:
        return '#48bb78'
    elif 'priority_tier' in row and row['priority_tier'] == 'Tier 1 - Critical':
        return '#e53e3e'
    elif 'priority_tier' in row and row['priority_tier'] == 'Tier 2 - High':
        return '#ed8936'
    else:
        return '#fc8181'

has_distance = 'distance_to_nearest_mshp_miles' in filtered_df.columns

if use_clustering:
    marker_cluster = MarkerCluster(name='Schools')
    for _, row in filtered_df.iterrows():
        distance_line = ""
        if has_distance and pd.notna(row.get('distance_to_nearest_mshp_miles')):
            distance_line = f"<p style='margin: 4px 0; color: #4a5568;'><b>Nearest MSHP:</b> {row['distance_to_nearest_mshp_miles']:.2f} mi</p>"

        popup_html = f"""
        <div style="font-family: Inter, sans-serif; width: 250px;">
            <h4 style="margin: 0 0 8px 0; color: #1a202c;">{row['school_name']}</h4>
            <p style="margin: 4px 0; color: #4a5568;"><b>DBN:</b> {row['dbn']}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Status:</b> 
                <span style="color: {'#48bb78' if row['has_mshp'] else '#e53e3e'};">
                    {'MSHP Covered' if row['has_mshp'] else 'Not Covered'}
                </span>
            </p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Enrollment:</b> {int(row['enrollment']):,}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Chronic Absenteeism:</b> {row['chronic_absenteeism_rate']:.1f}%</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Neighborhood:</b> {row['uhf_name']}</p>
            {distance_line}
            {'<p style="margin: 4px 0; color: #e53e3e;"><b>Priority:</b> ' + row.get('priority_tier', 'N/A') + '</p>' if not row['has_mshp'] else ''}
        </div>
        """
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='green' if row['has_mshp'] else 'red', icon='circle', prefix='fa')
        ).add_to(marker_cluster)
    marker_cluster.add_to(m)
else:
    for _, row in filtered_df.iterrows():
        distance_line = ""
        if has_distance and pd.notna(row.get('distance_to_nearest_mshp_miles')):
            distance_line = f"<p style='margin: 4px 0; color: #4a5568;'><b>Nearest MSHP:</b> {row['distance_to_nearest_mshp_miles']:.2f} mi</p>"

        popup_html = f"""
        <div style="font-family: Inter, sans-serif; width: 250px;">
            <h4 style="margin: 0 0 8px 0; color: #1a202c;">{row['school_name']}</h4>
            <p style="margin: 4px 0; color: #4a5568;"><b>DBN:</b> {row['dbn']}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Status:</b> 
                <span style="color: {'#48bb78' if row['has_mshp'] else '#e53e3e'};">
                    {'MSHP Covered' if row['has_mshp'] else 'Not Covered'}
                </span>
            </p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Enrollment:</b> {int(row['enrollment']):,}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Chronic Absenteeism:</b> {row['chronic_absenteeism_rate']:.1f}%</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Neighborhood:</b> {row['uhf_name']}</p>
            {distance_line}
            {'<p style="margin: 4px 0; color: #e53e3e;"><b>Priority:</b> ' + str(row.get("priority_tier", "N/A")) + '</p>' if not row['has_mshp'] else ''}
        </div>
        """
        radius = max(5, min(15, row['enrollment'] / 100))
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=get_marker_color(row),
            fill=True,
            fill_color=get_marker_color(row),
            fill_opacity=0.7,
            weight=2
        ).add_to(m)

deserts_df = pd.DataFrame()
if show_deserts and has_distance:
    deserts_df = filtered_df[
        (~filtered_df['has_mshp']) & (filtered_df['distance_to_nearest_mshp_miles'] >= desert_threshold)
    ].copy()
    for _, row in deserts_df.iterrows():
        radius = max(5, min(15, row['enrollment'] / 100))
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius + 6,
            color='#63b3ed',
            fill=False,
            weight=3,
            opacity=0.95,
        ).add_to(m)

folium.LayerControl().add_to(m)

map_state = st_folium(m, width=None, height=600, returned_objects=["last_object_clicked"])

clicked = (map_state or {}).get("last_object_clicked")
if isinstance(clicked, dict):
    lat = clicked.get("lat") or clicked.get("latitude")
    lon = clicked.get("lng") or clicked.get("lon") or clicked.get("longitude")
    if lat is not None and lon is not None and len(filtered_df) > 0:
        click_df = filtered_df.dropna(subset=['latitude', 'longitude']).copy()
        if len(click_df) == 0:
            click_df = filtered_df
        coords = click_df[['latitude', 'longitude']].to_numpy(dtype=float, copy=True)
        scaled = coords.copy()
        scaled[:, 0] *= 69.0
        scaled[:, 1] *= 53.0
        target = np.array([float(lat) * 69.0, float(lon) * 53.0])
        d = np.sqrt(np.sum((scaled - target) ** 2, axis=1))
        idx = int(np.argmin(d))
        nearest = click_df.iloc[idx]
        nearest_miles = float(d[idx])

        if nearest_miles <= 0.08:
            with st.expander("Selected School Details", expanded=True):
                cols = st.columns([2, 1, 1])
                with cols[0]:
                    st.markdown(f"**{nearest['school_name']}**")
                    st.caption(f"{nearest['dbn']} | {nearest.get('school_type', 'School')} | {nearest['uhf_name']}")
                with cols[1]:
                    st.metric("Enrollment", f"{int(nearest['enrollment']):,}")
                with cols[2]:
                    st.metric("MSHP Status", "Covered" if bool(nearest['has_mshp']) else "Not Covered")

                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.metric("Absenteeism", f"{nearest['chronic_absenteeism_rate']:.1f}%")
                with detail_cols[1]:
                    if 'asthma_ed_rate' in nearest:
                        st.metric("Asthma Rate", f"{nearest['asthma_ed_rate']:.1f}/10K")
                with detail_cols[2]:
                    if has_distance and pd.notna(nearest.get('distance_to_nearest_mshp_miles')):
                        st.metric("Nearest MSHP", f"{nearest['distance_to_nearest_mshp_miles']:.2f} mi")
                with detail_cols[3]:
                    if 'priority_tier' in nearest:
                        st.metric("Priority Tier", str(nearest.get('priority_tier', 'N/A')))

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Map Legend")
    st.markdown("""
    | Color | Meaning |
    |-------|---------|
    | Green | MSHP Covered School |
    | Red | Not Covered (Tier 1 Critical) |
    | Orange | Not Covered (Tier 2 High) |
    | Light Red | Not Covered (Tier 3-4) |
    | Blue Ring | Health Desert (far from any MSHP) |
    
    **Circle size** represents school enrollment.
    
    **Background shading** shows asthma ED visit rates by neighborhood (darker = higher burden).
    """)

with col2:
    st.markdown("### Schools Displayed")
    st.metric("Total on Map", len(filtered_df))
    st.metric("MSHP Schools", filtered_df['has_mshp'].sum())
    st.metric("Non-MSHP Schools", len(filtered_df) - filtered_df['has_mshp'].sum())
    if show_deserts and has_distance:
        st.metric("Health Deserts", len(deserts_df))

st.markdown("---")
st.markdown("""
<div class="author-footer">
    <div class="author-name">Yoel Y. Plutchok</div>
    <a href="https://github.com/yoelplutchok/mshp-coverage-gap" target="_blank">View Source Code on GitHub</a>
    <div style="color: #666666; font-size: 0.8rem; margin-top: 1rem;">
        Data sources: NYC DOE, NYC DOHMH, Montefiore Einstein
    </div>
</div>
""", unsafe_allow_html=True)
