"""
Page 4: Neighborhood Analysis
=============================
Deep dive into neighborhood-level health and coverage data.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Neighborhoods | MSHP Analysis",
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
    
    .neighborhood-header {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, get_neighborhood_summary
from utils.charts import create_scatter_health_coverage, apply_dark_theme, COLORS

df = load_schools_data()
neighborhood_summary = get_neighborhood_summary(df)

st.title("Neighborhood Analysis")
st.markdown("Deep dive into UHF neighborhood-level health burden and MSHP coverage.")

selected_neighborhood = st.selectbox(
    "Select a neighborhood to explore",
    options=neighborhood_summary['uhf_name'].tolist(),
    index=0
)

neigh_data = neighborhood_summary[neighborhood_summary['uhf_name'] == selected_neighborhood].iloc[0]
neigh_schools = df[df['uhf_name'] == selected_neighborhood]

st.markdown(f"""
<div class="neighborhood-header">
    <h2 style="color: #ffffff; margin: 0;">{selected_neighborhood}</h2>
    <p style="color: #e2e8f0; margin: 0.5rem 0 0 0;">
        {int(neigh_data['total_schools'])} schools | {int(neigh_data['total_students']):,} students
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("MSHP Coverage", f"{neigh_data['coverage_pct']:.0f}%")
with col2:
    st.metric("MSHP Schools", int(neigh_data['mshp_count']))
with col3:
    st.metric("Non-MSHP Schools", int(neigh_data['non_mshp_count']))
with col4:
    st.metric("Asthma Rate", f"{neigh_data['asthma_rate']:.0f}/10K")
with col5:
    st.metric("Avg Absenteeism", f"{neigh_data['avg_absenteeism']:.1f}%")

st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Coverage Breakdown")
    
    fig = go.Figure(data=[go.Pie(
        labels=['MSHP Covered', 'Not Covered'],
        values=[neigh_data['mshp_count'], neigh_data['non_mshp_count']],
        hole=0.6,
        marker_colors=[COLORS['mshp'], COLORS['non_mshp']],
        textinfo='value+percent',
        textfont=dict(size=14, color=COLORS['text']),
    )])
    
    fig.update_layout(
        showlegend=True,
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        annotations=[dict(
            text=f'{neigh_data["coverage_pct"]:.0f}%',
            x=0.5, y=0.5,
            font=dict(size=28, color=COLORS['text']),
            showarrow=False
        )]
    )
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("Schools in Neighborhood")
    
    mshp_schools = neigh_schools[neigh_schools['has_mshp']]
    non_mshp_schools = neigh_schools[~neigh_schools['has_mshp']]
    
    st.markdown("**MSHP Covered Schools:**")
    if len(mshp_schools) > 0:
        for _, school in mshp_schools.iterrows():
            st.markdown(f"- {school['school_name']} ({int(school['enrollment']):,} students)")
    else:
        st.markdown("*No MSHP schools in this neighborhood*")
    
    st.markdown("**Schools Without MSHP:**")
    if len(non_mshp_schools) > 0:
        top_non_mshp = non_mshp_schools.nlargest(5, 'priority_score') if 'priority_score' in non_mshp_schools.columns else non_mshp_schools.head(5)
        for _, school in top_non_mshp.iterrows():
            tier = school.get('priority_tier', 'N/A')
            st.markdown(f"- {school['school_name']} ({int(school['enrollment']):,} students) - {tier}")
        if len(non_mshp_schools) > 5:
            st.caption(f"...and {len(non_mshp_schools) - 5} more")
    else:
        st.markdown("*All schools have MSHP coverage*")

st.markdown("---")

st.subheader("Compared to Other Neighborhoods")

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('MSHP Coverage Rate', 'Asthma ED Rate', 'Avg Chronic Absenteeism'),
    horizontal_spacing=0.1
)

for i, (metric, title) in enumerate([
    ('coverage_pct', 'Coverage %'),
    ('asthma_rate', 'Asthma Rate'),
    ('avg_absenteeism', 'Absenteeism %')
], 1):
    sorted_df = neighborhood_summary.sort_values(metric)
    colors = [COLORS['primary'] if n == selected_neighborhood else COLORS['muted'] for n in sorted_df['uhf_name']]
    short_names = [n.split(' - ')[0] for n in sorted_df['uhf_name']]
    
    fig.add_trace(go.Bar(
        y=short_names,
        x=sorted_df[metric],
        orientation='h',
        marker_color=colors,
        showlegend=False,
        text=[f'{v:.0f}' for v in sorted_df[metric]],
        textposition='outside',
        textfont=dict(color=COLORS['text']),
    ), row=1, col=i)

fig.update_layout(height=400)
fig = apply_dark_theme(fig)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Health Burden vs Coverage Analysis")

fig = create_scatter_health_coverage(neighborhood_summary)
st.plotly_chart(fig, use_container_width=True)

if neigh_data['asthma_rate'] > neighborhood_summary['asthma_rate'].median():
    if neigh_data['coverage_pct'] < neighborhood_summary['coverage_pct'].median():
        st.error(f"""
        **Priority Area**: {selected_neighborhood} has **high health burden** 
        (asthma rate {neigh_data['asthma_rate']:.0f}/10K) but **low MSHP coverage** 
        ({neigh_data['coverage_pct']:.0f}%). This neighborhood should be prioritized for expansion.
        """)
    else:
        st.success(f"""
        **Well Targeted**: {selected_neighborhood} has high health burden 
        and relatively good MSHP coverage ({neigh_data['coverage_pct']:.0f}%). 
        MSHP is appropriately serving this high-need area.
        """)
else:
    if neigh_data['coverage_pct'] < neighborhood_summary['coverage_pct'].median():
        st.info(f"""
        **Monitor**: {selected_neighborhood} has lower health burden 
        but also lower MSHP coverage. While not urgent, expansion here 
        would improve equity.
        """)
    else:
        st.success(f"""
        **Good Coverage**: {selected_neighborhood} has adequate MSHP coverage 
        for its health burden level.
        """)

st.markdown("---")
st.subheader("All Schools in Neighborhood")

display_cols = ['school_name', 'dbn', 'mshp_status', 'enrollment', 'chronic_absenteeism_rate']
if 'school_type' in neigh_schools.columns:
    display_cols.insert(2, 'school_type')
if 'priority_tier' in neigh_schools.columns:
    display_cols.append('priority_tier')

display_df = neigh_schools[display_cols].copy()
display_df = display_df.rename(columns={
    'school_name': 'School',
    'dbn': 'DBN',
    'school_type': 'Type',
    'mshp_status': 'MSHP Status',
    'enrollment': 'Students',
    'chronic_absenteeism_rate': 'Absenteeism %',
    'priority_tier': 'Priority'
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div class="author-footer">
    <div class="author-name">Yoel Y. Plutchok</div>
    <a href="https://github.com/yoelplutchok/nyc-school-environmental-health" target="_blank">View Source Code on GitHub</a>
    <div style="color: #666666; font-size: 0.8rem; margin-top: 1rem;">
        Data sources: NYC DOE, NYC DOHMH, Montefiore Einstein
    </div>
</div>
""", unsafe_allow_html=True)
