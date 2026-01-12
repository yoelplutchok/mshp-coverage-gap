"""
MSHP Coverage Gap Analysis Dashboard
=====================================

An interactive dashboard for exploring Montefiore School Health Program
coverage gaps across Bronx public schools.

Run with: streamlit run app/Overview.py
"""
import streamlit as st

st.set_page_config(
    page_title="MSHP Coverage Gap Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - light theme matching localhost:8502
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', system-ui, sans-serif;
        background-color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0054a3 0%, #003366 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 84, 163, 0.15);
        color: #ffffff;
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.25rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #e3f2fd !important;
        font-size: 1.2rem;
        margin: 0.75rem 0 0 0;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1;
    }
    
    .metric-value.green { color: #27ae60 !important; }
    .metric-value.red { color: #e74c3c !important; }
    .metric-value.blue { color: #0054a3 !important; }
    .metric-value.amber { color: #f39c12 !important; }
    
    .metric-label {
        color: #444444;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .finding-box {
        background: #f8f9fa;
        border-left: 4px solid #0054a3;
        padding: 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        color: #1a1a2e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .finding-box.priority {
        border-left-color: #e74c3c;
    }
    
    .finding-box.success {
        border-left-color: #27ae60;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Force all text to be readable in light theme */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #333333;
    }
    
    .stApp h1, .stApp h2, .stApp h3 {
        color: #1a1a2e !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    
    [data-testid="stSidebar"] * {
        color: #333333 !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #0054a3 !important;
    }
    
    /* Page links in sidebar */
    [data-testid="stSidebar"] [data-testid="stPageLink"] span {
        color: #333333 !important;
    }
    
    /* Metric styling overrides */
    [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }
    
    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    .author-footer .author-name {
        font-weight: 700;
        color: #1a1a2e;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .author-footer a {
        color: #0054a3;
        text-decoration: none;
        font-weight: 500;
    }
    
    .author-footer a:hover {
        text-decoration: underline;
    }
    
    .author-footer div {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, get_neighborhood_summary
from utils.charts import create_coverage_donut, create_neighborhood_coverage_bar

@st.cache_data
def get_data():
    return load_schools_data()

df = get_data()

total_schools = len(df)
mshp_count = df['has_mshp'].sum()
non_mshp_count = total_schools - mshp_count
coverage_rate = mshp_count / total_schools * 100
total_students = df['enrollment'].sum()
mshp_students = df[df['has_mshp']]['enrollment'].sum()

# Header
st.markdown("""
<div class="main-header">
    <h1>MSHP Coverage Gap Analysis</h1>
    <p>Mapping school-based healthcare access across the Bronx</p>
</div>
""", unsafe_allow_html=True)

# Overview Section
st.markdown("""
### Project Overview
The **Montefiore School Health Program (MSHP)** is the largest school-based health center network in the United States, providing comprehensive primary, dental, and mental health services to students directly on campus. Despite its extensive reach, a significant number of Bronx public schools still lack on-site healthcare access, creating a "coverage gap" in neighborhoods with the highest health and social needs.

This dashboard provides a data-driven analysis of that gap. By integrating data from the NYC Department of Education, the Department of Health and Mental Hygiene, and Montefiore Einstein, we identify where healthcare needs are greatest and where MSHP expansion could have the most profound impact.

Our analysis focuses on three primary dimensions:
1. **Health Burden:** Specifically focusing on childhood asthma ED visit rates, a critical indicator of pediatric health need in the Bronx.
2. **Educational Impact:** Analyzing the relationship between healthcare access and chronic absenteeism.
3. **Health Equity:** Ensuring that expansion efforts prioritize the most vulnerable student populations and address geographic "health deserts."
""")

# Navigation at the top
st.markdown("### Explore the Data")

nav_row1 = st.columns(4)
nav_row2 = st.columns(3)

with nav_row1[0]:
    st.page_link("pages/1_Coverage_Map.py", label="Interactive Map")
    st.caption("Schools on map with health burden overlay")

with nav_row1[1]:
    st.page_link("pages/2_School_Explorer.py", label="School Explorer")
    st.caption("Search and filter all Bronx schools")

with nav_row1[2]:
    st.page_link("pages/3_Priority_Rankings.py", label="Priority Rankings")
    st.caption("Schools ranked by expansion priority")

with nav_row1[3]:
    st.page_link("pages/6_Expansion_Simulator.py", label="Expansion Simulator")
    st.caption("What-if scenarios for MSHP expansion")

with nav_row2[0]:
    st.page_link("pages/4_Neighborhoods.py", label="Neighborhoods")
    st.caption("Neighborhood-level analysis")

with nav_row2[1]:
    st.page_link("pages/5_Equity_Dashboard.py", label="Equity Dashboard")
    st.caption("Who is served vs left behind")

with nav_row2[2]:
    st.page_link("pages/7_Impact_Analysis.py", label="Impact and Outcomes")
    st.caption("Absenteeism and regression analysis")

st.markdown("---")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value blue">{total_schools}</div>
        <div class="metric-label">Bronx Schools</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value green">{mshp_count}</div>
        <div class="metric-label">MSHP Covered</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value red">{non_mshp_count}</div>
        <div class="metric-label">Not Covered</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value amber">{coverage_rate:.0f}%</div>
        <div class="metric-label">Coverage Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main content - two columns
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("### Coverage Overview")
    fig = create_coverage_donut(df)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True, key="main_coverage_donut")
    
    st.markdown("""
    <div class="finding-box">
        <strong>Key Finding:</strong> MSHP serves approximately 
        <strong>{:,}</strong> students ({:.0f}% of Bronx public school enrollment).
    </div>
    """.format(int(mshp_students), mshp_students/total_students*100), unsafe_allow_html=True)

with right_col:
    st.markdown("### Coverage by Neighborhood")
    fig = create_neighborhood_coverage_bar(df)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True, key="main_neighborhood_bar")

st.markdown("---")
st.markdown("### Priority Areas for Expansion")

neighborhood_summary = get_neighborhood_summary(df)
priority_neighborhoods = neighborhood_summary[neighborhood_summary['coverage_pct'] < 20]

if len(priority_neighborhoods) > 0:
    cols = st.columns(len(priority_neighborhoods))
    for i, (_, row) in enumerate(priority_neighborhoods.iterrows()):
        with cols[i]:
            short_name = row['uhf_name'].split(' - ')[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value red">{row['coverage_pct']:.0f}%</div>
                <div class="metric-label">{short_name}</div>
                <div style="color: #444444; font-size: 0.85rem; margin-top: 0.5rem; font-weight: 500;">
                    {int(row['non_mshp_count'])} schools without MSHP<br>
                    Asthma rate: {row['asthma_rate']:.0f}/10K
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="finding-box priority">
    <strong>Action Required:</strong> These neighborhoods have both 
    <strong>high health burden</strong> and <strong>low MSHP coverage</strong>. 
    Prioritize expansion here for maximum impact.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="author-footer">
    <div class="author-name">Yoel Y. Plutchok</div>
    <a href="https://github.com/yoelplutchok/mshp-coverage-gap" target="_blank">View Source Code on GitHub</a>
    <div style="color: #666666; font-size: 0.8rem; margin-top: 1rem;">
        Data sources: NYC DOE, NYC DOHMH, Montefiore Einstein
    </div>
</div>
""", unsafe_allow_html=True)
