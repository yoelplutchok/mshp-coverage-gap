"""
Page 3: Priority Rankings
=========================
Top priority schools for MSHP expansion with scoring breakdown.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Priority Rankings | MSHP Analysis",
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
    
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .rank-1 { background: #f1c40f; color: #1a202c; }
    .rank-2 { background: #bdc3c7; color: #1a202c; }
    .rank-3 { background: #e67e22; color: #1a202c; }
    .rank-other { background: #ecf0f1; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, load_expansion_scenarios
from utils.charts import create_priority_tier_chart, create_expansion_impact_chart, apply_dark_theme, COLORS

df = load_schools_data()
non_mshp = df[~df['has_mshp']].copy()

if 'priority_score' in non_mshp.columns:
    non_mshp = non_mshp.sort_values('priority_score', ascending=False).reset_index(drop=True)
    non_mshp['rank'] = range(1, len(non_mshp) + 1)

st.title("Priority Rankings")
st.markdown("Schools ranked by expansion priority based on health burden, absenteeism, and enrollment.")

col1, col2, col3, col4 = st.columns(4)

tier_counts = non_mshp['priority_tier'].value_counts() if 'priority_tier' in non_mshp.columns else {}

with col1:
    st.metric("Schools Without MSHP", len(non_mshp))
with col2:
    st.metric("Tier 1 - Critical", tier_counts.get('Tier 1 - Critical', 0))
with col3:
    st.metric("Tier 2 - High", tier_counts.get('Tier 2 - High', 0))
with col4:
    total_students = non_mshp['enrollment'].sum()
    st.metric("Students Without Access", f"{int(total_students):,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Top Priority Schools", "Expansion Scenarios", "Compare Schools"])

with tab1:
    top_n = st.slider("Show top N schools", min_value=10, max_value=50, value=25, step=5)
    
    top_schools = non_mshp.head(top_n)
    
    for i, row in top_schools.iterrows():
        rank = row.get('rank', i + 1)
        rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
        
        tier = row.get('priority_tier', 'N/A')
        tier_color = {
            'Tier 1 - Critical': '#ff4b4b',
            'Tier 2 - High': '#ff8c00',
            'Tier 3 - Moderate': '#ffd700',
            'Tier 4 - Lower': '#e2e8f0'
        }.get(tier, '#94a3b8')
        
        priority_score = row.get('priority_score', 0)
        
        with st.container():
            cols = st.columns([1, 6, 2, 2, 2])
            
            with cols[0]:
                st.markdown(f'<div class="rank-badge {rank_class}">{rank}</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"**{row['school_name']}**")
                st.caption(f"{row['dbn']} | {row.get('school_type', 'School')} | {row['uhf_name']}")
            
            with cols[2]:
                st.metric("Score", f"{priority_score:.1f}", label_visibility="collapsed")
                st.caption("Priority Score")
            
            with cols[3]:
                st.metric("Enrollment", f"{int(row['enrollment']):,}", label_visibility="collapsed")
                st.caption("Students")
            
            with cols[4]:
                st.markdown(f'<span style="color: {tier_color}; font-weight: 600;">{tier}</span>', unsafe_allow_html=True)
                st.caption("Tier")
            
            st.markdown("---")
    
    with st.expander("How Priority Scores Are Calculated"):
        st.markdown("""
        **Priority Score = Weighted combination of:**
        
        | Factor | Weight | Description |
        |--------|--------|-------------|
        | Asthma Burden | 40% | Neighborhood asthma ED visit rate (per 10K children) |
        | Chronic Absenteeism | 40% | School's chronic absenteeism rate |
        | Enrollment | 20% | Number of students (larger = more impact) |
        
        **Tier Classification:**
        - **Tier 1 - Critical**: Top 20% of priority scores
        - **Tier 2 - High**: 60th-80th percentile
        - **Tier 3 - Moderate**: 40th-60th percentile
        - **Tier 4 - Lower**: Bottom 40%
        """)

with tab2:
    st.subheader("Expansion Scenario Analysis")
    st.markdown("Compare different strategies for adding N more MSHP schools.")
    
    scenarios = load_expansion_scenarios()
    
    if scenarios['comparison'] is not None:
        fig = create_expansion_impact_chart(scenarios['comparison'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_schools = st.selectbox("Schools to add", options=[5, 10, 20], index=1)
            method = st.selectbox(
                "Selection method",
                options=['greedy', 'coverage_optimized', 'equity_constrained'],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            scenario_df = scenarios.get(n_schools)
            if scenario_df is not None:
                method_col = None
                if 'method' in scenario_df.columns:
                    method_col = 'method'
                elif 'selection_method' in scenario_df.columns:
                    method_col = 'selection_method'
                if method_col:
                    scenario_df = scenario_df[scenario_df[method_col] == method]
                
                st.markdown(f"### Recommended {n_schools} Schools ({method.replace('_', ' ').title()})")
                
                display_cols = ['school_name', 'dbn', 'uhf_name', 'enrollment']
                if 'expansion_score' in scenario_df.columns:
                    display_cols.append('expansion_score')
                
                if 'selection_rank' in scenario_df.columns:
                    scenario_df = scenario_df.sort_values('selection_rank')

                st.dataframe(scenario_df[display_cols].head(n_schools), use_container_width=True, hide_index=True)
            else:
                st.info("Scenario data not available. Run the optimal expansion analysis first.")
    else:
        st.info("Expansion scenario data not available. Run scripts/18_optimal_expansion.py to generate.")

with tab3:
    st.subheader("Compare Schools Side-by-Side")
    
    school_options = non_mshp['school_name'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        school1 = st.selectbox("Select first school", options=school_options, index=0)
    
    with col2:
        remaining_options = [s for s in school_options if s != school1]
        school2 = st.selectbox("Select second school", options=remaining_options, index=0)
    
    s1 = non_mshp[non_mshp['school_name'] == school1].iloc[0]
    s2 = non_mshp[non_mshp['school_name'] == school2].iloc[0]
    
    st.markdown("---")
    
    compare_col1, compare_col2 = st.columns(2)
    
    for col, school_data, name in [(compare_col1, s1, school1), (compare_col2, s2, school2)]:
        with col:
            st.markdown(f"### {name}")
            st.caption(f"{school_data['dbn']} | {school_data.get('school_type', 'School')}")
            
            st.markdown("---")
            
            st.metric("Priority Rank", f"#{int(school_data.get('rank', 0))}")
            st.metric("Priority Score", f"{school_data.get('priority_score', 0):.1f}")
            st.metric("Priority Tier", school_data.get('priority_tier', 'N/A'))
            
            st.markdown("---")
            
            st.metric("Enrollment", f"{int(school_data['enrollment']):,}")
            st.metric("Chronic Absenteeism", f"{school_data['chronic_absenteeism_rate']:.1f}%")
            st.metric("Neighborhood", school_data['uhf_name'].split(' - ')[0])
            st.metric("Asthma Rate", f"{school_data['asthma_ed_rate']:.1f}/10K")
    
    st.markdown("---")
    st.subheader("Visual Comparison")
    
    categories = ['Priority Score', 'Enrollment', 'Absenteeism', 'Asthma Rate', 'Poverty %']
    
    def normalize(val, min_val, max_val):
        if max_val == min_val:
            return 50
        return (val - min_val) / (max_val - min_val) * 100
    
    s1_values = [
        normalize(s1.get('priority_score', 0), non_mshp['priority_score'].min(), non_mshp['priority_score'].max()),
        normalize(s1['enrollment'], non_mshp['enrollment'].min(), non_mshp['enrollment'].max()),
        normalize(s1['chronic_absenteeism_rate'], non_mshp['chronic_absenteeism_rate'].min(), non_mshp['chronic_absenteeism_rate'].max()),
        normalize(s1['asthma_ed_rate'], df['asthma_ed_rate'].min(), df['asthma_ed_rate'].max()),
        normalize(s1.get('pct_poverty', 0.5) * 100, 0, 100),
    ]
    
    s2_values = [
        normalize(s2.get('priority_score', 0), non_mshp['priority_score'].min(), non_mshp['priority_score'].max()),
        normalize(s2['enrollment'], non_mshp['enrollment'].min(), non_mshp['enrollment'].max()),
        normalize(s2['chronic_absenteeism_rate'], non_mshp['chronic_absenteeism_rate'].min(), non_mshp['chronic_absenteeism_rate'].max()),
        normalize(s2['asthma_ed_rate'], df['asthma_ed_rate'].min(), df['asthma_ed_rate'].max()),
        normalize(s2.get('pct_poverty', 0.5) * 100, 0, 100),
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=s1_values + [s1_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=school1[:20] + '...' if len(school1) > 20 else school1,
        line_color=COLORS['primary'],
        fillcolor="rgba(99, 179, 237, 0.3)"
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=s2_values + [s2_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=school2[:20] + '...' if len(school2) > 20 else school2,
        line_color=COLORS['accent'],
        fillcolor="rgba(236, 201, 75, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(0,0,0,0)'),
        showlegend=True,
        height=400
    )
    
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

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
