"""
MSHP Coverage Gap Analysis Dashboard
=====================================

An interactive dashboard for exploring Montefiore School Health Program
coverage gaps across Bronx public schools.

Run with: streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="MSHP Coverage Gap Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - professional styling with readable text everywhere
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', system-ui, sans-serif;
        background-color: #1a1a2e;
    }
    
    .main-header {
        background: #252545;
        padding: 1.5rem 2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #4a5568;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: #e2e8f0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: #252545;
        padding: 1.25rem;
        border-radius: 4px;
        border: 1px solid #4a5568;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        line-height: 1;
    }
    
    .metric-value.green { color: #48bb78 !important; }
    .metric-value.red { color: #fc8181 !important; }
    .metric-value.blue { color: #63b3ed !important; }
    .metric-value.amber { color: #ecc94b !important; }
    
    .metric-label {
        color: #e2e8f0;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .finding-box {
        background: #252545;
        border-left: 3px solid #4299e1;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .finding-box.priority {
        border-left-color: #fc8181;
    }
    
    .finding-box.success {
        border-left-color: #48bb78;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Force all text to be readable */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
    }
    
    /* Make captions readable - aggressive targeting */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p,
    .stApp [data-testid="stCaptionContainer"] span,
    .stApp .stCaption,
    .stApp small,
    [data-testid="stCaptionContainer"] *,
    .stCaption,
    .stCaption p,
    p[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] p {
        color: #e2e8f0 !important;
    }
    
    /* Target caption text that follows page links */
    [data-testid="stPageLink"] + p,
    [data-testid="stPageLink"] ~ p[data-testid="stCaptionContainer"],
    .element-container p.st-emotion-cache-nahz7x,
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Override any gray text colors from Streamlit defaults */
    .st-emotion-cache-nahz7x,
    .st-emotion-cache-eczf16,
    .st-emotion-cache-1gulkj5,
    .st-emotion-cache-6q9sum,
    .st-emotion-cache-1wivap2 {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #16162a !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #90cdf4 !important;
    }
    
    /* Page links in sidebar */
    [data-testid="stSidebar"] [data-testid="stPageLink"] span {
        color: #ffffff !important;
    }
    
    /* Fix any blue text on blue background */
    .st-emotion-cache-1v0mbdj, 
    .st-emotion-cache-16idsys,
    .st-emotion-cache-ue6h4q {
        color: #ffffff !important;
    }
    
    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #4a5568;
        margin-top: 2rem;
    }
    
    .author-footer .author-name {
        font-weight: 600;
        color: #ffffff;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .author-footer a {
        color: #90cdf4;
        text-decoration: none;
    }
    
    .author-footer a:hover {
        text-decoration: underline;
    }
    
    .author-footer div {
        color: #e2e8f0 !important;
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
    st.subheader("Coverage Overview")
    fig = create_coverage_donut(df)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="finding-box">
        <strong>Key Finding:</strong> MSHP serves approximately 
        <strong>{:,}</strong> students ({:.0f}% of Bronx public school enrollment).
    </div>
    """.format(int(mshp_students), mshp_students/total_students*100), unsafe_allow_html=True)

with right_col:
    st.subheader("Coverage by Neighborhood")
    fig = create_neighborhood_coverage_bar(df)
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Priority Areas for Expansion")

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
                <div style="color: #e2e8f0; font-size: 0.8rem; margin-top: 0.5rem;">
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
    <div style="color: #e2e8f0; font-size: 0.8rem; margin-top: 1rem;">
        Data sources: NYC DOE, NYC DOHMH, Montefiore Einstein
    </div>
</div>
""", unsafe_allow_html=True)
