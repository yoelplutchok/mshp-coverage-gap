"""
Reusable chart components for the MSHP Coverage Gap Dashboard.
Uses Plotly for interactive charts.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Color palette - light theme matching localhost:8502
COLORS = {
    'mshp': '#27ae60',        # Success green
    'non_mshp': '#e74c3c',    # Danger red
    'accent': '#f39c12',      # Warning orange
    'primary': '#0054a3',     # Link blue from 8502
    'secondary': '#8e44ad',   # Purple
    'background': '#ffffff',  # White background
    'surface': '#f8f9fa',     # Light gray surface
    'text': '#1a1a2e',        # Dark navy text from 8502
    'muted': '#444444',       # Muted gray text
    'tier1': '#c0392b',       # Priority red
    'tier2': '#d35400',       # Priority orange
    'tier3': '#f1c40f',       # Priority yellow
    'tier4': '#7f8c8d',       # Neutral gray
}

TIER_COLORS = {
    'Tier 1 - Critical': COLORS['tier1'],
    'Tier 2 - High': COLORS['tier2'],
    'Tier 3 - Moderate': COLORS['tier3'],
    'Tier 4 - Lower': COLORS['tier4'],
}


def apply_dark_theme(fig):
    """Apply consistent light theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS['text'], family='Inter, system-ui, sans-serif'),
        title_font=dict(size=18, color=COLORS['text']),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            font=dict(color=COLORS['muted'])
        ),
        xaxis=dict(
            gridcolor='rgba(0, 0, 0, 0.05)',
            zerolinecolor='rgba(0, 0, 0, 0.1)',
            tickfont=dict(color=COLORS['muted']),
            titlefont=dict(color=COLORS['text'])
        ),
        yaxis=dict(
            gridcolor='rgba(0, 0, 0, 0.05)',
            zerolinecolor='rgba(0, 0, 0, 0.1)',
            tickfont=dict(color=COLORS['muted']),
            titlefont=dict(color=COLORS['text'])
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def create_coverage_donut(df: pd.DataFrame) -> go.Figure:
    """Create a donut chart showing MSHP coverage breakdown."""
    mshp_count = df['has_mshp'].sum()
    non_mshp_count = len(df) - mshp_count
    
    fig = go.Figure(data=[go.Pie(
        labels=['MSHP Covered', 'Not Covered'],
        values=[mshp_count, non_mshp_count],
        hole=0.6,
        marker_colors=[COLORS['mshp'], COLORS['non_mshp']],
        textinfo='label+percent',
        textfont=dict(size=14, color=COLORS['text']),
        hovertemplate='%{label}<br>%{value} schools<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            text=f'{mshp_count}<br>Schools',
            x=0.5, y=0.5,
            font=dict(size=20, color=COLORS['text']),
            showarrow=False
        )]
    )
    
    return apply_dark_theme(fig)


def create_neighborhood_coverage_bar(df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of coverage by neighborhood."""
    summary = df.groupby('uhf_name').agg({
        'has_mshp': ['sum', 'count']
    }).reset_index()
    summary.columns = ['uhf_name', 'mshp_count', 'total']
    summary['coverage_pct'] = summary['mshp_count'] / summary['total'] * 100
    summary = summary.sort_values('coverage_pct')
    
    # Shorten names for display
    summary['display_name'] = summary['uhf_name'].apply(
        lambda x: x.split(' - ')[0] if ' - ' in x else x
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=summary['display_name'],
        x=summary['coverage_pct'],
        orientation='h',
        marker_color=[COLORS['tier1'] if v < 20 else COLORS['mshp'] for v in summary['coverage_pct']],
        text=[f'{v:.0f}%' for v in summary['coverage_pct']],
        textposition='outside',
        textfont=dict(color=COLORS['text']),
        hovertemplate='%{y}<br>Coverage: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='MSHP Coverage by Neighborhood',
        xaxis_title='Coverage Rate (%)',
        yaxis_title='',
        height=350,
    )
    
    return apply_dark_theme(fig)


def create_absenteeism_comparison(df: pd.DataFrame) -> go.Figure:
    """Create box plot comparing absenteeism between MSHP and non-MSHP schools."""
    fig = go.Figure()
    
    for status, color in [('MSHP Covered', COLORS['mshp']), ('Not Covered', COLORS['non_mshp'])]:
        subset = df[df['mshp_status'] == status]
        fig.add_trace(go.Box(
            y=subset['chronic_absenteeism_rate'],
            name=status,
            marker_color=color,
            boxmean=True,
            hovertemplate='%{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Chronic Absenteeism Distribution',
        yaxis_title='Chronic Absenteeism Rate (%)',
        showlegend=True,
        height=400,
    )
    
    return apply_dark_theme(fig)


def create_priority_tier_chart(df: pd.DataFrame) -> go.Figure:
    """Create bar chart showing priority tier distribution."""
    if 'priority_tier' not in df.columns:
        return None
    
    tier_counts = df['priority_tier'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=tier_counts.index,
        y=tier_counts.values,
        marker_color=[TIER_COLORS.get(t, COLORS['muted']) for t in tier_counts.index],
        text=tier_counts.values,
        textposition='outside',
        textfont=dict(color=COLORS['text']),
        hovertemplate='%{x}<br>%{y} schools<extra></extra>'
    )])
    
    fig.update_layout(
        title='Non-MSHP Schools by Priority Tier',
        xaxis_title='',
        yaxis_title='Number of Schools',
        height=350,
    )
    
    return apply_dark_theme(fig)


def create_scatter_health_coverage(summary_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of health burden vs coverage by neighborhood."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=summary_df['asthma_rate'],
        y=summary_df['coverage_pct'],
        mode='markers+text',
        marker=dict(
            size=summary_df['total_schools'] * 2,
            color=summary_df['asthma_rate'],
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title='Asthma Rate'),
        ),
        text=summary_df['uhf_name'].apply(lambda x: x.split(' - ')[0]),
        textposition='top center',
        textfont=dict(color=COLORS['text'], size=10),
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Asthma Rate: %{x:.1f}<br>'
            'Coverage: %{y:.1f}%<br>'
            '<extra></extra>'
        )
    ))
    
    # Add quadrant lines
    median_asthma = summary_df['asthma_rate'].median()
    median_coverage = summary_df['coverage_pct'].median()
    
    fig.add_hline(y=median_coverage, line_dash='dash', line_color=COLORS['muted'], opacity=0.5)
    fig.add_vline(x=median_asthma, line_dash='dash', line_color=COLORS['muted'], opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper',
                       text='Low Burden, High Coverage', showarrow=False,
                       font=dict(size=10, color=COLORS['muted']))
    fig.add_annotation(x=0.98, y=0.98, xref='paper', yref='paper',
                       text='High Burden, High Coverage (Well-Targeted)', showarrow=False,
                       font=dict(size=10, color=COLORS['mshp']))
    fig.add_annotation(x=0.02, y=0.02, xref='paper', yref='paper',
                       text='Low Burden, Low Coverage', showarrow=False,
                       font=dict(size=10, color=COLORS['muted']))
    fig.add_annotation(x=0.98, y=0.02, xref='paper', yref='paper',
                       text='High Burden, Low Coverage (Priority)', showarrow=False,
                       font=dict(size=10, color=COLORS['tier1']))
    
    fig.update_layout(
        title='Health Burden vs MSHP Coverage',
        xaxis_title='Asthma ED Visit Rate (per 10K)',
        yaxis_title='MSHP Coverage (%)',
        height=450,
    )
    
    return apply_dark_theme(fig)


def create_demographic_comparison(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing demographics."""
    mshp = df[df['has_mshp']]
    non_mshp = df[~df['has_mshp']]
    
    metrics = ['pct_poverty', 'pct_black', 'pct_hispanic', 'pct_ell', 'pct_swd']
    labels = ['Poverty', 'Black', 'Hispanic', 'ELL', 'SWD']
    
    mshp_means = [mshp[m].mean() * 100 if mshp[m].max() <= 1 else mshp[m].mean() for m in metrics]
    non_mshp_means = [non_mshp[m].mean() * 100 if non_mshp[m].max() <= 1 else non_mshp[m].mean() for m in metrics]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='MSHP Covered',
        x=labels,
        y=mshp_means,
        marker_color=COLORS['mshp'],
        text=[f'{v:.1f}%' for v in mshp_means],
        textposition='outside',
    ))
    
    fig.add_trace(go.Bar(
        name='Not Covered',
        x=labels,
        y=non_mshp_means,
        marker_color=COLORS['non_mshp'],
        text=[f'{v:.1f}%' for v in non_mshp_means],
        textposition='outside',
    ))
    
    fig.update_layout(
        title='Demographics: MSHP vs Non-MSHP Schools',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=400,
    )
    
    return apply_dark_theme(fig)


def create_school_type_coverage(df: pd.DataFrame) -> go.Figure:
    """Create coverage rate by school type."""
    if 'school_type' not in df.columns:
        return None
    
    type_summary = df.groupby('school_type').agg({
        'has_mshp': ['sum', 'count', 'mean']
    }).reset_index()
    type_summary.columns = ['school_type', 'mshp_count', 'total', 'coverage_rate']
    type_summary['coverage_pct'] = type_summary['coverage_rate'] * 100
    type_summary = type_summary.sort_values('coverage_pct', ascending=False)
    
    fig = go.Figure(data=[go.Bar(
        x=type_summary['school_type'],
        y=type_summary['coverage_pct'],
        marker_color=COLORS['primary'],
        text=[f'{v:.0f}%' for v in type_summary['coverage_pct']],
        textposition='outside',
        textfont=dict(color=COLORS['text']),
        hovertemplate='%{x}<br>Coverage: %{y:.1f}%<br><extra></extra>'
    )])
    
    fig.update_layout(
        title='MSHP Coverage by School Type',
        xaxis_title='',
        yaxis_title='Coverage Rate (%)',
        height=350,
    )
    
    return apply_dark_theme(fig)


def create_expansion_impact_chart(scenarios_df: pd.DataFrame) -> go.Figure:
    """Create chart showing impact of different expansion scenarios."""
    if scenarios_df is None:
        return None
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Students Reached', 'Coverage Rate'),
        horizontal_spacing=0.15
    )
    
    for method, color in [('greedy', COLORS['primary']), 
                           ('coverage_optimized', COLORS['secondary']),
                           ('equity_constrained', COLORS['accent'])]:
        method_data = scenarios_df[scenarios_df['method'] == method]
        
        fig.add_trace(go.Scatter(
            x=method_data['n_schools'],
            y=method_data['students_reached'],
            mode='lines+markers',
            name=method.replace('_', ' ').title(),
            line=dict(color=color),
            marker=dict(size=10),
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=method_data['n_schools'],
            y=method_data['new_coverage_rate_pct'],
            mode='lines+markers',
            name=method.replace('_', ' ').title(),
            line=dict(color=color),
            marker=dict(size=10),
            showlegend=False,
        ), row=1, col=2)
    
    fig.update_xaxes(title_text='Schools Added', row=1, col=1)
    fig.update_xaxes(title_text='Schools Added', row=1, col=2)
    fig.update_yaxes(title_text='Students', row=1, col=1)
    fig.update_yaxes(title_text='Coverage (%)', row=1, col=2)
    
    fig.update_layout(
        title='Expansion Scenario Comparison',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return apply_dark_theme(fig)

