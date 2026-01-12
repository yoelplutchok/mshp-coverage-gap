"""
Page 7: Impact and Outcomes
===========================
Outcome-focused analysis (absenteeism + regression summaries).
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Impact and Outcomes | MSHP Analysis",
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
    load_regression_results,
    load_regression_summary_text,
)
from utils.charts import create_absenteeism_comparison, apply_dark_theme, COLORS

df = load_schools_data()

st.title("Impact and Outcomes")
st.markdown(
    "This page focuses on student outcomes (chronic absenteeism) and what we can "
    "infer about MSHP's association with those outcomes."
)

st.subheader("Absenteeism Distribution: MSHP vs Non-MSHP")
fig = create_absenteeism_comparison(df)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Dose-Response: Years with MSHP vs Absenteeism (MSHP schools only)")
if 'years_with_mshp' in df.columns:
    dose_df = df[df['has_mshp']].copy()
    dose_df = dose_df[pd.notna(dose_df['years_with_mshp']) & pd.notna(dose_df['chronic_absenteeism_rate'])]

    if len(dose_df) >= 10:
        color_col = 'asthma_ed_rate' if 'asthma_ed_rate' in dose_df.columns else None
        fig2 = px.scatter(
            dose_df,
            x='years_with_mshp',
            y='chronic_absenteeism_rate',
            size='enrollment' if 'enrollment' in dose_df.columns else None,
            color=color_col,
            hover_data=['school_name', 'dbn'] if 'school_name' in dose_df.columns else ['dbn'],
            labels={
                'years_with_mshp': 'Years with MSHP',
                'chronic_absenteeism_rate': 'Chronic Absenteeism (%)',
                'asthma_ed_rate': 'Asthma ED Rate (per 10K)',
            },
        )
        fig2.update_layout(height=450, title="Dose-Response (descriptive)")
        fig2 = apply_dark_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        x = dose_df['years_with_mshp'].to_numpy(dtype=float)
        y = dose_df['chronic_absenteeism_rate'].to_numpy(dtype=float)
        if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            st.caption(f"Descriptive correlation: r = {r:.2f} (not causal).")
    else:
        st.info("Not enough MSHP schools with `years_with_mshp` to plot a dose-response.")
else:
    st.info(
        "Dose-response requires a `years_with_mshp` column in `bronx_schools_full.csv`. "
        "If you have MSHP start years, add/compute it in the pipeline."
    )

st.markdown("---")

st.subheader("Regression Analysis")
st.markdown("""
**What is this?** Regression analysis estimates whether MSHP coverage is associated with 
lower chronic absenteeism, while controlling for other factors like poverty, school type, 
and neighborhood health burden. A negative MSHP coefficient would suggest schools with MSHP 
have lower absenteeism rates.
""")

reg_df = load_regression_results()

if reg_df is None or len(reg_df) == 0:
    st.info(
        "Regression results have not been generated yet. "
        "Run `python scripts/11_enhanced_analysis.py` to generate them."
    )
else:
    full = reg_df[reg_df['model'] == 'model3_full']
    if len(full) == 1:
        row = full.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Full model MSHP coef", f"{row['mshp_coefficient']:.3f}")
        with col2:
            st.metric("p-value", f"{row['mshp_pvalue']:.4f}")
        with col3:
            st.metric("R-squared", f"{row['r_squared']:.3f}")

    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    summary_text = load_regression_summary_text()
    if summary_text:
        with st.expander("Full model output (text)"):
            st.code(summary_text)

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
