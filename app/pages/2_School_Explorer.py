"""
Page 2: School Explorer
=======================
Searchable, filterable table of all Bronx schools.
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="School Explorer | MSHP Analysis",
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
    
    .school-card {
        background: #252545;
        padding: 1.5rem;
        border-radius: 4px;
        border: 1px solid #4a5568;
        margin-bottom: 1rem;
    }
    
    .school-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .school-dbn {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-covered { background: rgba(72, 187, 120, 0.2); color: #48bb78; }
    .status-not-covered { background: rgba(252, 129, 129, 0.2); color: #fc8181; }
</style>
""", unsafe_allow_html=True)

from utils.data_loader import load_schools_data, get_school_types, get_districts

df = load_schools_data()

st.title("School Explorer")
st.markdown("Search and filter all 370 Bronx public schools.")

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    search = st.text_input(
        "Search schools",
        placeholder="Type school name or DBN...",
        label_visibility="collapsed"
    )

with col2:
    mshp_filter = st.selectbox("MSHP Status", options=['All', 'MSHP Covered', 'Not Covered'], index=0)

with col3:
    school_types = ['All'] + get_school_types(df)
    type_filter = st.selectbox("School Type", options=school_types, index=0)

with col4:
    neighborhoods = ['All'] + sorted(df['uhf_name'].dropna().unique().tolist())
    neighborhood_filter = st.selectbox("Neighborhood", options=neighborhoods, index=0)

filtered_df = df.copy()

if search:
    search_lower = search.lower()
    filtered_df = filtered_df[
        filtered_df['school_name'].str.lower().str.contains(search_lower, na=False) |
        filtered_df['dbn'].str.lower().str.contains(search_lower, na=False)
    ]

if mshp_filter == 'MSHP Covered':
    filtered_df = filtered_df[filtered_df['has_mshp']]
elif mshp_filter == 'Not Covered':
    filtered_df = filtered_df[~filtered_df['has_mshp']]

if type_filter != 'All' and 'school_type' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['school_type'] == type_filter]

if neighborhood_filter != 'All':
    filtered_df = filtered_df[filtered_df['uhf_name'] == neighborhood_filter]

st.markdown("---")
sort_col1, sort_col2 = st.columns([1, 3])

with sort_col1:
    sort_by = st.selectbox(
        "Sort by",
        options=[
            'School Name',
            'Enrollment (High to Low)',
            'Enrollment (Low to High)',
            'Chronic Absenteeism (High to Low)',
            'Chronic Absenteeism (Low to High)',
            'Priority Score (High to Low)'
        ],
        index=0
    )

if sort_by == 'School Name':
    filtered_df = filtered_df.sort_values('school_name')
elif sort_by == 'Enrollment (High to Low)':
    filtered_df = filtered_df.sort_values('enrollment', ascending=False)
elif sort_by == 'Enrollment (Low to High)':
    filtered_df = filtered_df.sort_values('enrollment', ascending=True)
elif sort_by == 'Chronic Absenteeism (High to Low)':
    filtered_df = filtered_df.sort_values('chronic_absenteeism_rate', ascending=False)
elif sort_by == 'Chronic Absenteeism (Low to High)':
    filtered_df = filtered_df.sort_values('chronic_absenteeism_rate', ascending=True)
elif sort_by == 'Priority Score (High to Low)' and 'priority_score' in filtered_df.columns:
    filtered_df = filtered_df.sort_values('priority_score', ascending=False)

st.markdown(f"**{len(filtered_df)}** schools found")

view_mode = st.radio("View Mode", options=['Table', 'Cards'], horizontal=True, label_visibility="collapsed")

if view_mode == 'Table':
    display_cols = ['school_name', 'dbn', 'mshp_status', 'enrollment', 'chronic_absenteeism_rate', 'uhf_name']
    
    if 'school_type' in filtered_df.columns:
        display_cols.insert(3, 'school_type')
    
    if 'priority_tier' in filtered_df.columns:
        display_cols.append('priority_tier')
    
    display_df = filtered_df[display_cols].copy()
    
    column_names = {
        'school_name': 'School Name',
        'dbn': 'DBN',
        'mshp_status': 'MSHP Status',
        'school_type': 'Type',
        'enrollment': 'Enrollment',
        'chronic_absenteeism_rate': 'Chronic Absenteeism %',
        'uhf_name': 'Neighborhood',
        'priority_tier': 'Priority Tier'
    }
    display_df = display_df.rename(columns=column_names)
    
    if 'Chronic Absenteeism %' in display_df.columns:
        display_df['Chronic Absenteeism %'] = display_df['Chronic Absenteeism %'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="mshp_schools_filtered.csv",
        mime="text/csv"
    )

else:
    for _, row in filtered_df.head(50).iterrows():
        status_class = "status-covered" if row['has_mshp'] else "status-not-covered"
        status_text = "MSHP Covered" if row['has_mshp'] else "Not Covered"
        
        priority_html = ""
        if not row['has_mshp'] and 'priority_tier' in row:
            priority_html = f'<span style="color: #ecc94b; margin-left: 10px;">{row["priority_tier"]}</span>'
        
        with st.container():
            st.markdown(f"""
            <div class="school-card">
                <div class="school-name">{row['school_name']}</div>
                <div class="school-dbn">{row['dbn']} | {row.get('school_type', 'School')}</div>
                <div style="margin-top: 1rem;">
                    <span class="status-badge {status_class}">{status_text}</span>
                    {priority_html}
                </div>
                <div style="margin-top: 1rem; display: flex; gap: 2rem;">
                    <div>
                        <div style="color: #e2e8f0; font-size: 0.8rem;">Enrollment</div>
                        <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">{int(row['enrollment']):,}</div>
                    </div>
                    <div>
                        <div style="color: #e2e8f0; font-size: 0.8rem;">Chronic Absenteeism</div>
                        <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">{row['chronic_absenteeism_rate']:.1f}%</div>
                    </div>
                    <div>
                        <div style="color: #e2e8f0; font-size: 0.8rem;">Neighborhood</div>
                        <div style="color: #ffffff; font-size: 1.1rem; font-weight: 600;">{row['uhf_name'].split(' - ')[0]}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if len(filtered_df) > 50:
        st.info(f"Showing first 50 of {len(filtered_df)} schools. Use table view for full list.")
