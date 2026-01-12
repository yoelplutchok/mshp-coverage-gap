"""
Page 6: Expansion Simulator
===========================
Interactive "what-if" scenarios for MSHP expansion.
"""
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

st.set_page_config(
    page_title="Expansion Simulator | MSHP Analysis",
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

from utils.data_loader import (
    load_schools_data,
    load_expansion_scenarios,
    load_uhf_boundaries,
    get_neighborhood_summary,
)
from utils.charts import create_expansion_impact_chart, COLORS

df = load_schools_data()
neighborhood_summary = get_neighborhood_summary(df)
scenarios = load_expansion_scenarios()

st.title("Expansion Simulator")
st.markdown("Choose an expansion strategy and see which schools are recommended — plus how coverage changes.")

st.sidebar.header("Scenario Controls")
n_schools = st.sidebar.selectbox("Schools to add", options=[5, 10, 20], index=1)
method = st.sidebar.selectbox(
    "Selection strategy",
    options=['greedy', 'coverage_optimized', 'equity_constrained'],
    format_func=lambda x: x.replace('_', ' ').title(),
)

st.sidebar.header("Map Layers")
show_choropleth = st.sidebar.checkbox("Show asthma burden choropleth", value=True)
show_existing_mshp = st.sidebar.checkbox("Show existing MSHP schools", value=True)
show_all_non_mshp = st.sidebar.checkbox("Show all non-MSHP schools (busy)", value=False)

show_deserts = st.sidebar.checkbox("Highlight current health deserts", value=True)
desert_threshold = 1.0
if show_deserts:
    desert_threshold = st.sidebar.slider(
        "Desert threshold (miles to nearest MSHP)",
        min_value=0.25, max_value=2.0, value=1.0, step=0.25,
    )

scenario_df = scenarios.get(n_schools)
if scenario_df is None or len(scenario_df) == 0:
    st.info(
        "Expansion scenario tables were not found.\n\n"
        "Generate them by running:\n"
        "- `python scripts/18_optimal_expansion.py`\n\n"
        "This should create files in `outputs/tables/`."
    )
    st.stop()

method_col = None
if 'method' in scenario_df.columns:
    method_col = 'method'
elif 'selection_method' in scenario_df.columns:
    method_col = 'selection_method'
if method_col:
    scenario_df = scenario_df[scenario_df[method_col] == method].copy()

if 'selection_rank' in scenario_df.columns:
    scenario_df = scenario_df.sort_values('selection_rank').head(n_schools)
elif 'expansion_score' in scenario_df.columns:
    scenario_df = scenario_df.sort_values('expansion_score', ascending=False).head(n_schools)
else:
    scenario_df = scenario_df.head(n_schools)

total_schools = len(df)
current_mshp = int(df['has_mshp'].sum())
current_coverage = current_mshp / total_schools * 100 if total_schools else 0
new_mshp = current_mshp + len(scenario_df)
new_coverage = new_mshp / total_schools * 100 if total_schools else 0

students_reached = int(scenario_df['enrollment'].sum()) if 'enrollment' in scenario_df.columns else 0
neighborhoods_covered = int(scenario_df['uhf_name'].nunique()) if 'uhf_name' in scenario_df.columns else 0

metric1, metric2, metric3, metric4 = st.columns(4)
with metric1:
    st.metric("Current Coverage", f"{current_coverage:.1f}%")
with metric2:
    st.metric("Coverage After Expansion", f"{new_coverage:.1f}%")
with metric3:
    st.metric("Students Reached", f"{students_reached:,}")
with metric4:
    st.metric("Neighborhoods Covered", f"{neighborhoods_covered}")

comparison_df = scenarios.get('comparison')
if comparison_df is not None and len(comparison_df) > 0:
    with st.expander("Compare all strategies", expanded=False):
        fig = create_expansion_impact_chart(comparison_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

coords_cols = ['dbn', 'latitude', 'longitude', 'has_mshp', 'priority_tier', 'priority_score']
coords_cols = [c for c in coords_cols if c in df.columns]
scenario_map_df = scenario_df.merge(df[coords_cols], on='dbn', how='left', suffixes=('', '_base'))

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("Recommended Expansion Map")

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
                fill_opacity=0.35,
                line_opacity=0.6,
                legend_name='Asthma ED Visits (per 10K)',
                name='Asthma Burden',
            ).add_to(m)
        except Exception as e:
            st.warning(f"Could not load choropleth: {e}")

    if show_existing_mshp:
        mshp_df = df[df['has_mshp']].copy()
        for _, row in mshp_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=COLORS['mshp'],
                fill=True,
                fill_color=COLORS['mshp'],
                fill_opacity=0.6,
                weight=1,
                tooltip="MSHP",
            ).add_to(m)

    if show_all_non_mshp:
        non_mshp_df = df[~df['has_mshp']].copy()
        for _, row in non_mshp_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=COLORS['non_mshp'],
                fill=True,
                fill_color=COLORS['non_mshp'],
                fill_opacity=0.25,
                weight=1,
                tooltip="No MSHP",
            ).add_to(m)

    if show_deserts and 'distance_to_nearest_mshp_miles' in df.columns:
        deserts = df[(~df['has_mshp']) & (df['distance_to_nearest_mshp_miles'] >= desert_threshold)].copy()
        for _, row in deserts.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=10,
                color='#63b3ed',
                fill=False,
                weight=2,
                opacity=0.9,
                tooltip=f"Health desert (>={desert_threshold:.2f} mi)",
            ).add_to(m)

    for _, row in scenario_map_df.iterrows():
        if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
            continue

        enrollment_val = row.get('enrollment', 0)
        enrollment_str = f"{int(enrollment_val):,}" if pd.notna(enrollment_val) else "N/A"
        
        popup_html = f"""
        <div style="font-family: Inter, sans-serif; width: 260px;">
            <h4 style="margin: 0 0 8px 0; color: #1a202c;">{row.get('school_name', 'School')}</h4>
            <p style="margin: 4px 0; color: #4a5568;"><b>DBN:</b> {row.get('dbn', '')}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Neighborhood:</b> {row.get('uhf_name', '')}</p>
            <p style="margin: 4px 0; color: #4a5568;"><b>Enrollment:</b> {enrollment_str}</p>
            {f"<p style='margin: 4px 0; color: #4a5568;'><b>Expansion score:</b> {row.get('expansion_score', 0):.2f}</p>" if 'expansion_score' in row else ""}
        </div>
        """

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=320),
            icon=folium.Icon(color='blue', icon='plus', prefix='fa'),
            tooltip="Recommended",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=650, returned_objects=[])

with right_col:
    st.subheader("Recommended Schools")

    display_cols = ['selection_rank', 'school_name', 'dbn', 'uhf_name', 'enrollment', 'expansion_score']
    display_cols = [c for c in display_cols if c in scenario_df.columns]
    if 'selection_rank' in scenario_df.columns:
        scenario_df = scenario_df.sort_values('selection_rank')

    st.dataframe(scenario_df[display_cols], use_container_width=True, hide_index=True, height=520)

    csv = scenario_df.to_csv(index=False)
    st.download_button(
        label="Download this recommendation (CSV)",
        data=csv,
        file_name=f"mshp_expansion_{n_schools}_{method}.csv",
        mime="text/csv",
        use_container_width=True,
    )

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
