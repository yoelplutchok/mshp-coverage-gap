"""
Page 5: Equity Dashboard
========================
Demographic breakdowns showing who is served vs not served.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Equity Dashboard | MSHP Analysis",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { 
        font-family: 'Inter', system-ui, sans-serif; 
        background-color: #1a1a2e;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ensure readable text */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 { 
        color: #ffffff !important; 
    }
    [data-testid="stSidebar"] { background: #16162a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    [data-testid="stSidebar"] a { color: #90cdf4 !important; }
    
    /* Make captions readable - aggressive targeting */
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
    .st-emotion-cache-1wivap2 { color: #e2e8f0 !important; }
    
    .equity-card {
        background: #252545;
        padding: 1.25rem;
        border-radius: 4px;
        border: 1px solid #4a5568;
        text-align: center;
        height: 100%;
    }
    
    .equity-stat {
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    .finding-box {
        background: #252545;
        border-left: 3px solid #4299e1;
        padding: 1rem 1.25rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .finding-box.warning {
        border-left-color: #ecc94b;
    }
    
    .finding-box.critical {
        border-left-color: #fc8181;
    }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, load_equity_data
from utils.charts import create_demographic_comparison, create_school_type_coverage, apply_dark_theme, COLORS

df = load_schools_data()
equity_data = load_equity_data()

mshp = df[df['has_mshp']]
non_mshp = df[~df['has_mshp']]

st.title("Equity Dashboard")
st.markdown("Understanding who is served and who is left behind.")

st.subheader("Key Equity Metrics")

col1, col2, col3, col4 = st.columns(4)

high_risk_count = len(equity_data.get('high_risk_uncovered_schools', pd.DataFrame()))

with col1:
    st.markdown(f"""
    <div class="equity-card">
        <div style="color: #e2e8f0;">Students with Access</div>
        <div class="equity-stat" style="color: #48bb78;">{mshp['enrollment'].sum():,.0f}</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">{mshp['enrollment'].sum() / df['enrollment'].sum() * 100:.0f}% of enrollment</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="equity-card">
        <div style="color: #e2e8f0;">Students Without Access</div>
        <div class="equity-stat" style="color: #fc8181;">{non_mshp['enrollment'].sum():,.0f}</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">{non_mshp['enrollment'].sum() / df['enrollment'].sum() * 100:.0f}% of enrollment</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="equity-card">
        <div style="color: #e2e8f0;">High-Risk Schools Uncovered</div>
        <div class="equity-stat" style="color: #ecc94b;">{high_risk_count}</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">3+ risk factors, no MSHP</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    if 'school_type' in df.columns:
        elem = df[df['school_type'] == 'Elementary']
        elem_coverage = elem['has_mshp'].mean() * 100
    else:
        elem_coverage = 0
    st.markdown(f"""
    <div class="equity-card">
        <div style="color: #e2e8f0;">Elementary Coverage</div>
        <div class="equity-stat" style="color: #fc8181;">{elem_coverage:.0f}%</div>
        <div style="color: #e2e8f0; font-size: 0.9rem;">Lowest of all school types</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "School Type", "Intersectional Risk", "Disparities"])

with tab1:
    st.subheader("Demographic Comparison: MSHP vs Non-MSHP Schools")
    
    fig = create_demographic_comparison(df)
    st.plotly_chart(fig, use_container_width=True)
    
    mshp_poverty = mshp['pct_poverty'].mean() * 100
    non_mshp_poverty = non_mshp['pct_poverty'].mean() * 100
    
    if mshp_poverty >= non_mshp_poverty:
        st.markdown(f"""
        <div class="finding-box">
            <strong>Equity-Focused Targeting:</strong> MSHP schools serve slightly 
            <strong>higher poverty</strong> populations ({mshp_poverty:.0f}% vs {non_mshp_poverty:.0f}%), 
            indicating the program is appropriately targeting high-need schools.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="finding-box warning">
            <strong>Potential Equity Gap:</strong> Non-MSHP schools have 
            <strong>higher poverty rates</strong> ({non_mshp_poverty:.0f}% vs {mshp_poverty:.0f}%), 
            suggesting expansion should prioritize these underserved populations.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Detailed Demographic Breakdown")
    
    metrics = [
        ('pct_poverty', 'Poverty Rate', 100),
        ('pct_black', 'Black Students', 100),
        ('pct_hispanic', 'Hispanic Students', 100),
        ('pct_ell', 'English Language Learners', 100),
        ('pct_swd', 'Students with Disabilities', 100),
    ]
    
    demo_data = []
    for col, label, multiplier in metrics:
        if col in df.columns:
            mshp_val = mshp[col].mean() * multiplier
            non_mshp_val = non_mshp[col].mean() * multiplier
            diff = mshp_val - non_mshp_val
            demo_data.append({
                'Metric': label,
                'MSHP Schools': f'{mshp_val:.1f}%',
                'Non-MSHP Schools': f'{non_mshp_val:.1f}%',
                'Difference': f'{diff:+.1f}pp'
            })
    
    st.dataframe(pd.DataFrame(demo_data), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Coverage by School Type")
    
    fig = create_school_type_coverage(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    if 'school_type' in df.columns:
        type_summary = df.groupby('school_type').agg({
            'has_mshp': ['sum', 'count', 'mean'],
            'enrollment': 'sum'
        }).reset_index()
        type_summary.columns = ['School Type', 'MSHP Count', 'Total Schools', 'Coverage Rate', 'Total Students']
        type_summary['Coverage %'] = type_summary['Coverage Rate'] * 100
        type_summary['Non-MSHP'] = type_summary['Total Schools'] - type_summary['MSHP Count']
        
        display_df = type_summary[['School Type', 'Total Schools', 'MSHP Count', 'Non-MSHP', 'Coverage %', 'Total Students']].copy()
        display_df['Coverage %'] = display_df['Coverage %'].apply(lambda x: f'{x:.1f}%')
        display_df['Total Students'] = display_df['Total Students'].apply(lambda x: f'{int(x):,}')
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="finding-box critical">
            <strong>Critical Gap:</strong> Elementary schools have significantly lower 
            MSHP coverage than high schools. Young children may have less access to 
            school-based healthcare despite potentially greater need.
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.subheader("Intersectional Risk Analysis")
    st.markdown("""
    Schools with **multiple risk factors** represent the highest-need population.
    Risk factors include: high poverty, high absenteeism, high asthma burden, and high special needs.
    """)
    
    risk_df = equity_data.get('equity_intersectional_risk')
    
    if risk_df is not None:
        fig = go.Figure()
        
        main_rows = risk_df[risk_df['risk_factor_count'] != '3+ (High Risk)']
        
        fig.add_trace(go.Bar(
            x=main_rows['risk_factor_count'].astype(str) + ' factors',
            y=main_rows['total_schools'],
            marker_color=[COLORS['muted'], COLORS['tier3'], COLORS['tier2'], COLORS['tier1'], COLORS['tier1']],
            text=main_rows['total_schools'],
            textposition='outside',
        ))
        
        fig.update_layout(
            title='Distribution of Schools by Number of Risk Factors',
            xaxis_title='Number of Risk Factors',
            yaxis_title='Number of Schools',
            height=400
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### MSHP Coverage by Risk Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                main_rows['risk_count_numeric'] = pd.to_numeric(main_rows['risk_factor_count'], errors='coerce')
                low_risk = main_rows[main_rows['risk_count_numeric'].isin([0, 1])]
                low_risk_coverage = low_risk['coverage_rate'].mean() * 100 if len(low_risk) > 0 else 0
            except:
                low_risk_coverage = 0
            st.metric("Low Risk Coverage (0-1 factors)", f"{low_risk_coverage:.1f}%")
        
        with col2:
            try:
                high_risk = main_rows[main_rows['risk_count_numeric'] >= 3]
                high_risk_coverage = high_risk['coverage_rate'].mean() * 100 if len(high_risk) > 0 else 0
            except:
                high_risk_coverage = 0
            st.metric("High Risk Coverage (3+ factors)", f"{high_risk_coverage:.1f}%")
        
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    else:
        st.info("Intersectional risk data not available. Run scripts/15_equity_analysis.py to generate.")
    
    st.markdown("---")
    st.subheader("High-Risk Schools Without MSHP")
    
    high_risk_df = equity_data.get('high_risk_uncovered_schools')
    
    if high_risk_df is not None and len(high_risk_df) > 0:
        st.warning(f"**{len(high_risk_df)} schools** have 3+ risk factors but no MSHP coverage.")
        
        display_cols = ['school_name', 'dbn', 'uhf_name', 'enrollment']
        if 'priority_score' in high_risk_df.columns:
            display_cols.append('priority_score')
        
        st.dataframe(high_risk_df[display_cols].head(10), use_container_width=True, hide_index=True)
        
        if len(high_risk_df) > 10:
            with st.expander(f"View all {len(high_risk_df)} high-risk schools"):
                st.dataframe(high_risk_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("High-risk uncovered schools data not available.")

with tab4:
    st.subheader("Demographic Disparity Analysis")
    
    disparities_df = equity_data.get('equity_demographic_disparities')
    
    if disparities_df is not None:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='MSHP Schools',
            x=disparities_df['metric'],
            y=disparities_df['mshp_mean'],
            marker_color=COLORS['mshp'],
            text=[f'{v:.1f}' for v in disparities_df['mshp_mean']],
            textposition='outside',
        ))
        
        fig.add_trace(go.Bar(
            name='Non-MSHP Schools',
            x=disparities_df['metric'],
            y=disparities_df['non_mshp_mean'],
            marker_color=COLORS['non_mshp'],
            text=[f'{v:.1f}' for v in disparities_df['non_mshp_mean']],
            textposition='outside',
        ))
        
        fig.update_layout(
            title='Mean Values by MSHP Status',
            barmode='group',
            xaxis_title='',
            yaxis_title='Value',
            height=400,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Statistical Significance")
        
        display_df = disparities_df[['metric', 'mshp_mean', 'non_mshp_mean', 'difference', 'p_value', 'significant']].copy()
        display_df['Significant?'] = display_df['significant'].map({True: 'Yes', False: 'No'})
        display_df = display_df.rename(columns={
            'metric': 'Metric',
            'mshp_mean': 'MSHP Mean',
            'non_mshp_mean': 'Non-MSHP Mean',
            'difference': 'Difference',
            'p_value': 'P-Value'
        })
        
        st.dataframe(display_df[['Metric', 'MSHP Mean', 'Non-MSHP Mean', 'Difference', 'P-Value', 'Significant?']], 
                    use_container_width=True, hide_index=True)
        
        sig_count = disparities_df['significant'].sum()
        if sig_count == 0:
            st.markdown("""
            <div class="finding-box">
                <strong>Key Finding:</strong> No statistically significant demographic 
                differences between MSHP and non-MSHP schools. This suggests the program 
                is reaching a representative cross-section of Bronx schools.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="finding-box warning">
                <strong>Key Finding:</strong> {sig_count} demographic metric(s) show 
                statistically significant differences between MSHP and non-MSHP schools.
                Review the data above to understand these disparities.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Disparity analysis data not available. Run scripts/15_equity_analysis.py to generate.")

st.markdown("---")
st.subheader("Equity Recommendations")

st.markdown("""
Based on this analysis, we recommend:

1. **Prioritize Elementary Schools**: With only ~6% coverage, elementary schools represent the largest gap in MSHP reach.

2. **Target High-Risk, High-Need Schools**: Focus expansion on the 127+ schools with multiple risk factors that currently lack coverage.

3. **Address Geographic Disparities**: Neighborhoods like Hunts Point and Crotona have high health burden but low coverage.

4. **Maintain Equity Focus**: Current MSHP targeting appears equitable by demographics. Future expansion should maintain this balance.
""")
