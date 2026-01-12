#!/usr/bin/env python3
"""
Script 17: Equity & Health Burden Visualizations

Input:
    - data/processed/bronx_schools_full.csv
    - outputs/tables/equity_*.csv
    - outputs/tables/health_*.csv

Output:
    - outputs/figures/equity_demographic_disparities.png
    - outputs/figures/equity_intersectional_risk.png
    - outputs/figures/equity_by_school_type.png
    - outputs/figures/equity_geographic_gaps.png
    - outputs/figures/health_burden_composite_map.png
    - outputs/figures/health_coverage_mismatch.png

Creates compelling visualizations for equity and health burden analysis.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import PROCESSED_DIR, TABLES_DIR, FIGURES_DIR
from mshp_gap.logging_utils import get_logger

logger = get_logger("17_equity_visualizations")

# Set style - professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Color palette
COLORS = {
    'mshp': '#2E7D32',      # Green
    'non_mshp': '#C62828',  # Red
    'accent': '#1565C0',    # Blue
    'warning': '#FF8F00',   # Amber
    'neutral': '#757575',   # Gray
    'priority': '#B71C1C',  # Dark red
}


def load_data():
    """Load all required data files."""
    df = pd.read_csv(PROCESSED_DIR / 'bronx_schools_full.csv')
    print(f"Loaded {len(df)} schools")
    return df


def create_demographic_disparities_chart(df):
    """
    Create visualization showing demographic differences between MSHP and non-MSHP schools.
    """
    print("Creating demographic disparities chart...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Equity Analysis: Who is Served by MSHP?', fontsize=16, fontweight='bold', y=1.02)
    
    # Demographic metrics to visualize
    metrics = [
        ('pct_poverty', 'Poverty Rate', '%'),
        ('pct_black', 'Black Students', '%'),
        ('pct_hispanic', 'Hispanic Students', '%'),
        ('pct_ell', 'English Language Learners', '%'),
        ('pct_swd', 'Students with Disabilities', '%'),
        ('chronic_absenteeism_rate', 'Chronic Absenteeism', '%'),
        ('asthma_ed_rate', 'Asthma ED Rate', 'per 10K'),
        ('enrollment', 'Enrollment', 'students'),
    ]
    
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    for ax, (col, title, unit) in zip(axes.flatten(), metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        
        mshp_vals = mshp[col].dropna()
        non_mshp_vals = non_mshp[col].dropna()
        
        # Multiply by 100 for percentages stored as decimals
        if col.startswith('pct_') and df[col].max() <= 1:
            mshp_vals = mshp_vals * 100
            non_mshp_vals = non_mshp_vals * 100
        
        # Create violin plot with embedded box plot
        parts = ax.violinplot([mshp_vals, non_mshp_vals], positions=[0, 1], 
                              showmeans=True, showmedians=True, widths=0.7)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([COLORS['mshp'], COLORS['non_mshp']][i])
            pc.set_alpha(0.6)
        
        # Add mean annotations
        ax.annotate(f'{mshp_vals.mean():.1f}', xy=(0, mshp_vals.mean()),
                   xytext=(0.15, mshp_vals.mean()), fontsize=9, fontweight='bold',
                   color=COLORS['mshp'])
        ax.annotate(f'{non_mshp_vals.mean():.1f}', xy=(1, non_mshp_vals.mean()),
                   xytext=(1.15, non_mshp_vals.mean()), fontsize=9, fontweight='bold',
                   color=COLORS['non_mshp'])
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['MSHP\nSchools', 'Non-MSHP\nSchools'])
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(unit)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['mshp'], alpha=0.6, label='MSHP Schools'),
        mpatches.Patch(facecolor=COLORS['non_mshp'], alpha=0.6, label='Non-MSHP Schools'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'equity_demographic_disparities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'equity_demographic_disparities.png'}")


def create_intersectional_risk_chart(df):
    """
    Visualize MSHP coverage by number of risk factors.
    """
    print("Creating intersectional risk chart...")
    
    # Calculate risk factors
    df_risk = df.copy()
    
    poverty_threshold = 0.80
    absenteeism_threshold = 35.0
    asthma_median = df['asthma_ed_rate'].median()
    swd_median = df['pct_swd'].median()
    
    df_risk['risk_high_poverty'] = df_risk['pct_poverty'] > poverty_threshold
    df_risk['risk_high_absenteeism'] = df_risk['chronic_absenteeism_rate'] > absenteeism_threshold
    df_risk['risk_high_asthma'] = df_risk['asthma_ed_rate'] > asthma_median
    df_risk['risk_high_swd'] = df_risk['pct_swd'] > swd_median
    
    risk_cols = ['risk_high_poverty', 'risk_high_absenteeism', 'risk_high_asthma', 'risk_high_swd']
    df_risk['risk_count'] = df_risk[risk_cols].sum(axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Intersectional Risk Analysis: MSHP Coverage by Vulnerability', 
                 fontsize=14, fontweight='bold')
    
    # 1. Stacked bar chart by risk level
    ax1 = axes[0]
    
    risk_summary = df_risk.groupby('risk_count')['has_mshp'].agg(['sum', 'count']).reset_index()
    risk_summary.columns = ['risk_count', 'mshp_count', 'total']
    risk_summary['non_mshp_count'] = risk_summary['total'] - risk_summary['mshp_count']
    risk_summary['coverage_pct'] = risk_summary['mshp_count'] / risk_summary['total'] * 100
    
    x = risk_summary['risk_count']
    bar_width = 0.6
    
    ax1.bar(x, risk_summary['mshp_count'], bar_width, label='MSHP Covered', 
            color=COLORS['mshp'], alpha=0.8)
    ax1.bar(x, risk_summary['non_mshp_count'], bar_width, bottom=risk_summary['mshp_count'],
            label='Not Covered', color=COLORS['non_mshp'], alpha=0.8)
    
    # Add coverage percentages
    for i, row in risk_summary.iterrows():
        ax1.annotate(f'{row["coverage_pct"]:.0f}%', 
                    xy=(row['risk_count'], row['total'] + 5),
                    ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Number of Risk Factors')
    ax1.set_ylabel('Number of Schools')
    ax1.set_title('School Count by Risk Level')
    ax1.legend()
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['0\n(Low Risk)', '1', '2', '3', '4\n(High Risk)'])
    
    # 2. Coverage rate trend
    ax2 = axes[1]
    
    ax2.plot(risk_summary['risk_count'], risk_summary['coverage_pct'], 
             marker='o', linewidth=2, markersize=10, color=COLORS['accent'])
    ax2.fill_between(risk_summary['risk_count'], risk_summary['coverage_pct'], 
                     alpha=0.3, color=COLORS['accent'])
    
    ax2.axhline(23, linestyle='--', color=COLORS['neutral'], 
                label=f'Overall Coverage (23%)')
    
    ax2.set_xlabel('Number of Risk Factors')
    ax2.set_ylabel('MSHP Coverage Rate (%)')
    ax2.set_title('Coverage Rate by Vulnerability Level')
    ax2.set_ylim(0, 50)
    ax2.legend()
    ax2.set_xticks(range(5))
    
    # 3. High-risk school breakdown
    ax3 = axes[2]
    
    high_risk = df_risk[df_risk['risk_count'] >= 3]
    high_risk_mshp = high_risk['has_mshp'].sum()
    high_risk_non = len(high_risk) - high_risk_mshp
    
    sizes = [high_risk_mshp, high_risk_non]
    labels = [f'MSHP Covered\n({high_risk_mshp} schools)', 
              f'NOT Covered\n({high_risk_non} schools)']
    colors_pie = [COLORS['mshp'], COLORS['non_mshp']]
    explode = (0, 0.05)
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                        explode=explode, autopct='%1.0f%%',
                                        pctdistance=0.6, labeldistance=1.15)
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax3.set_title(f'High-Risk Schools (3+ Factors)\nTotal: {len(high_risk)} schools')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'equity_intersectional_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'equity_intersectional_risk.png'}")


def create_school_type_chart(df):
    """
    Visualize MSHP coverage by school type.
    """
    print("Creating school type coverage chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('MSHP Coverage by School Type', fontsize=14, fontweight='bold')
    
    # Aggregate by school type
    type_summary = df.groupby('school_type').agg({
        'dbn': 'count',
        'has_mshp': 'sum',
        'enrollment': 'sum',
    }).reset_index()
    type_summary.columns = ['school_type', 'total_schools', 'mshp_count', 'total_enrollment']
    type_summary['non_mshp'] = type_summary['total_schools'] - type_summary['mshp_count']
    type_summary['coverage_pct'] = type_summary['mshp_count'] / type_summary['total_schools'] * 100
    type_summary = type_summary.sort_values('coverage_pct', ascending=True)
    
    # 1. Horizontal stacked bar
    ax1 = axes[0]
    
    y = range(len(type_summary))
    
    ax1.barh(y, type_summary['mshp_count'], color=COLORS['mshp'], 
             label='MSHP Covered', alpha=0.8)
    ax1.barh(y, type_summary['non_mshp'], left=type_summary['mshp_count'],
             color=COLORS['non_mshp'], label='Not Covered', alpha=0.8)
    
    # Add coverage percentages
    for i, row in type_summary.iterrows():
        ax1.annotate(f'{row["coverage_pct"]:.0f}%', 
                    xy=(row['total_schools'] + 5, list(type_summary['school_type']).index(row['school_type'])),
                    va='center', fontsize=10, fontweight='bold')
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(type_summary['school_type'])
    ax1.set_xlabel('Number of Schools')
    ax1.set_title('Schools by Type and MSHP Status')
    ax1.legend(loc='lower right')
    
    # 2. Students covered bar chart
    ax2 = axes[1]
    
    # Calculate students by MSHP status for each type
    student_data = []
    for school_type in type_summary['school_type']:
        subset = df[df['school_type'] == school_type]
        mshp_students = subset[subset['has_mshp'] == True]['enrollment'].sum()
        non_mshp_students = subset[subset['has_mshp'] == False]['enrollment'].sum()
        student_data.append({
            'type': school_type,
            'mshp_students': mshp_students,
            'non_mshp_students': non_mshp_students,
            'total': mshp_students + non_mshp_students
        })
    
    student_df = pd.DataFrame(student_data)
    student_df = student_df.sort_values('total', ascending=True)
    
    y2 = range(len(student_df))
    
    ax2.barh(y2, student_df['mshp_students'], color=COLORS['mshp'], 
             label='Students with MSHP Access', alpha=0.8)
    ax2.barh(y2, student_df['non_mshp_students'], left=student_df['mshp_students'],
             color=COLORS['non_mshp'], label='Students without MSHP Access', alpha=0.8)
    
    ax2.set_yticks(y2)
    ax2.set_yticklabels(student_df['type'])
    ax2.set_xlabel('Number of Students')
    ax2.set_title('Students by School Type and MSHP Access')
    ax2.legend(loc='lower right')
    
    # Format with thousands separator
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'equity_by_school_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'equity_by_school_type.png'}")


def create_geographic_equity_chart(df):
    """
    Visualize geographic equity (health burden vs coverage by neighborhood).
    """
    print("Creating geographic equity chart...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Geographic Equity: Health Burden vs. MSHP Coverage', 
                 fontsize=14, fontweight='bold')
    
    # Aggregate to neighborhood
    neighborhood = df.groupby('uhf_name').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'asthma_ed_rate': 'first',
        'health_burden_composite': 'first',
        'enrollment': 'sum',
        'pct_poverty': 'mean',
    }).round(3)
    
    neighborhood.columns = ['schools', 'mshp_count', 'coverage_rate', 
                           'asthma_rate', 'health_burden', 'students', 'poverty']
    neighborhood = neighborhood.reset_index()
    neighborhood['coverage_pct'] = neighborhood['coverage_rate'] * 100
    
    # 1. Scatter plot: Health Burden vs Coverage
    ax1 = axes[0]
    
    scatter = ax1.scatter(neighborhood['health_burden'], neighborhood['coverage_pct'],
                         s=neighborhood['students']/100, c=neighborhood['poverty'],
                         cmap='YlOrRd', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add quadrant lines
    health_median = neighborhood['health_burden'].median()
    coverage_median = neighborhood['coverage_pct'].median()
    ax1.axvline(health_median, linestyle='--', color='gray', alpha=0.5)
    ax1.axhline(coverage_median, linestyle='--', color='gray', alpha=0.5)
    
    # Annotate neighborhoods
    for _, row in neighborhood.iterrows():
        # Shorten neighborhood names
        name = row['uhf_name'].split(' - ')[0][:12]
        ax1.annotate(name, (row['health_burden'], row['coverage_pct']),
                    fontsize=8, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points')
    
    # Add quadrant labels
    ax1.text(neighborhood['health_burden'].max() * 0.95, 
             neighborhood['coverage_pct'].max() * 0.95,
             'Good\nTargeting', ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax1.text(neighborhood['health_burden'].max() * 0.95, 
             neighborhood['coverage_pct'].min() * 1.2,
             'PRIORITY', ha='right', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
    
    ax1.set_xlabel('Health Burden Score')
    ax1.set_ylabel('MSHP Coverage (%)')
    ax1.set_title('Health Burden vs. Coverage\n(size = enrollment, color = poverty)')
    plt.colorbar(scatter, ax=ax1, label='Poverty Rate')
    
    # 2. Bar chart: Coverage gap by neighborhood
    ax2 = axes[1]
    
    # Calculate gap score
    neighborhood['gap_score'] = (
        neighborhood['health_burden'].rank(pct=True) * 100 -
        neighborhood['coverage_pct'].rank(pct=True) * 100
    )
    neighborhood = neighborhood.sort_values('gap_score', ascending=False)
    
    colors = [COLORS['priority'] if gap > 0 else COLORS['mshp'] 
              for gap in neighborhood['gap_score']]
    
    y = range(len(neighborhood))
    ax2.barh(y, neighborhood['gap_score'], color=colors, alpha=0.8)
    
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(neighborhood['uhf_name'])
    ax2.set_xlabel('Gap Score (+ = underserved)')
    ax2.set_title('Coverage Gap by Neighborhood\n(Health Burden Rank - Coverage Rank)')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['priority'], alpha=0.8, label='Underserved'),
        mpatches.Patch(facecolor=COLORS['mshp'], alpha=0.8, label='Adequately Served'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    # 3. Summary table
    ax3 = axes[2]
    ax3.axis('off')
    
    # Prepare table data
    table_data = neighborhood[['uhf_name', 'schools', 'mshp_count', 'coverage_pct', 'health_burden']].copy()
    table_data['coverage_pct'] = table_data['coverage_pct'].round(0).astype(int).astype(str) + '%'
    table_data['health_burden'] = table_data['health_burden'].round(1)
    table_data.columns = ['Neighborhood', 'Schools', 'MSHP', 'Coverage', 'Health Burden']
    
    table = ax3.table(cellText=table_data.values,
                      colLabels=table_data.columns,
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#E3F2FD')
        table[(0, i)].set_text_props(fontweight='bold')
    
    ax3.set_title('Neighborhood Summary', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'equity_geographic_gaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'equity_geographic_gaps.png'}")


def create_health_coverage_mismatch_chart(df):
    """
    Create health-coverage mismatch visualization (quadrant analysis).
    """
    print("Creating health-coverage mismatch chart...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Health Burden - MSHP Coverage Mismatch Analysis', 
                 fontsize=14, fontweight='bold')
    
    # Aggregate to neighborhood
    neighborhood = df.groupby('uhf_name').agg({
        'has_mshp': 'mean',
        'health_burden_composite': 'first',
        'asthma_ed_rate': 'first',
        'enrollment': 'sum',
    }).reset_index()
    neighborhood['coverage_pct'] = neighborhood['has_mshp'] * 100
    
    # 1. Quadrant chart
    ax1 = axes[0]
    
    # Calculate medians for quadrant lines
    health_median = neighborhood['health_burden_composite'].median()
    coverage_median = neighborhood['coverage_pct'].median()
    
    # Assign colors by quadrant
    def get_quadrant_color(row):
        if row['health_burden_composite'] >= health_median and row['coverage_pct'] < coverage_median:
            return COLORS['priority']  # High burden, low coverage = PRIORITY
        elif row['health_burden_composite'] >= health_median:
            return COLORS['mshp']  # High burden, high coverage = Good
        elif row['coverage_pct'] < coverage_median:
            return COLORS['warning']  # Low burden, low coverage = Monitor
        else:
            return COLORS['neutral']  # Low burden, high coverage = Review
    
    colors = [get_quadrant_color(row) for _, row in neighborhood.iterrows()]
    
    scatter = ax1.scatter(neighborhood['health_burden_composite'], 
                         neighborhood['coverage_pct'],
                         s=neighborhood['enrollment']/80,
                         c=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add quadrant lines
    ax1.axvline(health_median, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.axhline(coverage_median, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    
    # Annotate points
    for _, row in neighborhood.iterrows():
        name = row['uhf_name'].split(' - ')[0][:10]
        ax1.annotate(name, (row['health_burden_composite'], row['coverage_pct']),
                    fontsize=9, ha='left',
                    xytext=(5, 3), textcoords='offset points')
    
    # Label quadrants
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    
    ax1.text(xlim[1]*0.9, ylim[1]*0.9, 'Good\nTargeting', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(xlim[1]*0.9, ylim[0]+5, '⚠️ PRIORITY', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))
    ax1.text(xlim[0]+5, ylim[1]*0.9, 'Review', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax1.text(xlim[0]+5, ylim[0]+5, 'Monitor', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax1.set_xlabel('Health Burden Score')
    ax1.set_ylabel('MSHP Coverage (%)')
    ax1.set_title('Quadrant Analysis\n(size = student enrollment)')
    
    # 2. Mismatch score bar chart
    ax2 = axes[1]
    
    # Calculate mismatch score
    neighborhood['burden_rank'] = neighborhood['health_burden_composite'].rank(pct=True) * 100
    neighborhood['coverage_rank'] = neighborhood['coverage_pct'].rank(pct=True) * 100
    neighborhood['mismatch'] = neighborhood['burden_rank'] - neighborhood['coverage_rank']
    neighborhood = neighborhood.sort_values('mismatch', ascending=False)
    
    colors_bar = [COLORS['priority'] if m > 0 else COLORS['mshp'] 
                  for m in neighborhood['mismatch']]
    
    y = range(len(neighborhood))
    bars = ax2.barh(y, neighborhood['mismatch'], color=colors_bar, alpha=0.8, edgecolor='black')
    
    ax2.axvline(0, color='black', linewidth=1.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(neighborhood['uhf_name'])
    ax2.set_xlabel('Mismatch Score')
    ax2.set_title('Under/Over-Served Score\n(+ = underserved relative to need)')
    
    # Add value labels
    for i, (idx, row) in enumerate(neighborhood.iterrows()):
        ax2.annotate(f'{row["mismatch"]:+.0f}', 
                    xy=(row['mismatch'] + (2 if row['mismatch'] > 0 else -2), i),
                    va='center', ha='left' if row['mismatch'] > 0 else 'right',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'health_coverage_mismatch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'health_coverage_mismatch.png'}")


def create_summary_infographic(df):
    """
    Create a summary infographic with key equity findings.
    """
    print("Creating summary infographic...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('MSHP Coverage Gap Analysis: Key Equity Findings', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Calculate key metrics
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    total_schools = len(df)
    mshp_count = len(mshp)
    coverage_rate = mshp_count / total_schools * 100
    
    total_students = df['enrollment'].sum()
    mshp_students = mshp['enrollment'].sum()
    student_coverage = mshp_students / total_students * 100 if total_students > 0 else 0
    
    # 1. Big number: Coverage rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.6, f'{coverage_rate:.0f}%', fontsize=48, fontweight='bold',
             ha='center', va='center', color=COLORS['accent'])
    ax1.text(0.5, 0.2, f'Schools Covered\n({mshp_count}/{total_schools})', 
             fontsize=12, ha='center', va='center')
    ax1.axis('off')
    ax1.set_title('School Coverage', fontsize=12, fontweight='bold')
    
    # 2. Big number: Student coverage
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.6, f'{student_coverage:.0f}%', fontsize=48, fontweight='bold',
             ha='center', va='center', color=COLORS['mshp'])
    ax2.text(0.5, 0.2, f'Students with Access\n({mshp_students:,.0f}/{total_students:,.0f})', 
             fontsize=12, ha='center', va='center')
    ax2.axis('off')
    ax2.set_title('Student Access', fontsize=12, fontweight='bold')
    
    # 3. High-risk schools (3+ of 4 risk factors - consistent with equity analysis)
    df_risk = df.copy()
    poverty_threshold = 0.80
    absenteeism_threshold = 35.0
    asthma_median = df['asthma_ed_rate'].median()
    swd_median = df['pct_swd'].median()
    
    df_risk['risk_high_poverty'] = df_risk['pct_poverty'] > poverty_threshold
    df_risk['risk_high_absenteeism'] = df_risk['chronic_absenteeism_rate'] > absenteeism_threshold
    df_risk['risk_high_asthma'] = df_risk['asthma_ed_rate'] > asthma_median
    df_risk['risk_high_swd'] = df_risk['pct_swd'] > swd_median
    
    risk_cols = ['risk_high_poverty', 'risk_high_absenteeism', 'risk_high_asthma', 'risk_high_swd']
    df_risk['risk_count'] = df_risk[risk_cols].sum(axis=1)
    df_risk['is_high_risk'] = df_risk['risk_count'] >= 3
    
    high_risk = df_risk[df_risk['is_high_risk'] == True]
    high_risk_uncovered = high_risk[high_risk['has_mshp'] == False]
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, f'{len(high_risk_uncovered)}', fontsize=48, fontweight='bold',
             ha='center', va='center', color=COLORS['priority'])
    ax3.text(0.5, 0.2, f'High-Risk Schools\nWithout MSHP', 
             fontsize=12, ha='center', va='center')
    ax3.axis('off')
    ax3.set_title('Equity Gap', fontsize=12, fontweight='bold')
    
    # 4. Coverage by neighborhood (mini bar chart)
    ax4 = fig.add_subplot(gs[0, 3])
    
    neighborhood = df.groupby('uhf_name')['has_mshp'].mean() * 100
    neighborhood = neighborhood.sort_values()
    
    colors_neigh = [COLORS['priority'] if v < 20 else COLORS['mshp'] for v in neighborhood.values]
    ax4.barh(range(len(neighborhood)), neighborhood.values, color=colors_neigh, alpha=0.8)
    ax4.set_yticks(range(len(neighborhood)))
    ax4.set_yticklabels([n.split(' - ')[0][:15] for n in neighborhood.index], fontsize=8)
    ax4.set_xlabel('Coverage %')
    ax4.set_title('By Neighborhood', fontsize=12, fontweight='bold')
    
    # 5. Poverty comparison
    ax5 = fig.add_subplot(gs[1, :2])
    
    poverty_bins = pd.cut(df['pct_poverty'], bins=[0, 0.6, 0.8, 0.9, 1.0], 
                          labels=['<60%', '60-80%', '80-90%', '90%+'])
    poverty_coverage = df.groupby(poverty_bins, observed=True)['has_mshp'].agg(['mean', 'count'])
    poverty_coverage['coverage_pct'] = poverty_coverage['mean'] * 100
    
    bars = ax5.bar(range(len(poverty_coverage)), poverty_coverage['coverage_pct'], 
                   color=COLORS['accent'], alpha=0.8)
    ax5.set_xticks(range(len(poverty_coverage)))
    ax5.set_xticklabels(poverty_coverage.index)
    ax5.set_ylabel('MSHP Coverage %')
    ax5.set_xlabel('School Poverty Rate')
    ax5.set_title('MSHP Coverage by Poverty Level', fontsize=12, fontweight='bold')
    
    for bar, (idx, row) in zip(bars, poverty_coverage.iterrows()):
        ax5.annotate(f'{row["coverage_pct"]:.0f}%\n(n={row["count"]:.0f})', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    
    # 6. School type comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    
    type_coverage = df.groupby('school_type')['has_mshp'].agg(['mean', 'count'])
    type_coverage['coverage_pct'] = type_coverage['mean'] * 100
    type_coverage = type_coverage.sort_values('coverage_pct', ascending=True)
    
    bars = ax6.barh(range(len(type_coverage)), type_coverage['coverage_pct'], 
                    color=COLORS['accent'], alpha=0.8)
    ax6.set_yticks(range(len(type_coverage)))
    ax6.set_yticklabels(type_coverage.index)
    ax6.set_xlabel('MSHP Coverage %')
    ax6.set_title('MSHP Coverage by School Type', fontsize=12, fontweight='bold')
    
    for bar, (idx, row) in zip(bars, type_coverage.iterrows()):
        ax6.annotate(f'{row["coverage_pct"]:.0f}%', 
                    xy=(bar.get_width() + 1, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=9, fontweight='bold')
    
    # 7. Key findings text box
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    findings_text = """
    KEY EQUITY FINDINGS
    
    ✓ MSHP reaches 23% of Bronx schools (84/370), serving approximately {:.0f}% of students
    ✓ MSHP schools have HIGHER average poverty rates ({:.0f}% vs {:.0f}%), indicating equity-focused targeting
    ✓ {} high-risk schools with multiple vulnerability factors remain without MSHP coverage
    ✓ Coverage varies significantly by neighborhood: highest in {} ({:.0f}%), lowest in {} ({:.0f}%)
    ✓ High schools have highest coverage ({:.0f}%), while elementary schools have lower coverage
    
    RECOMMENDATION: Prioritize expansion to high-risk, high-poverty schools in underserved neighborhoods
    """.format(
        student_coverage,
        mshp['pct_poverty'].mean() * 100, non_mshp['pct_poverty'].mean() * 100,
        len(high_risk_uncovered),
        neighborhood.idxmax().split(' - ')[0], neighborhood.max(),
        neighborhood.idxmin().split(' - ')[0], neighborhood.min(),
        type_coverage['coverage_pct'].max()
    )
    
    ax7.text(0.5, 0.5, findings_text, fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8),
             family='monospace')
    
    plt.savefig(FIGURES_DIR / 'equity_summary_infographic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'equity_summary_infographic.png'}")


def main():
    """Generate all equity and health burden visualizations."""
    print("=" * 70)
    print("EQUITY & HEALTH BURDEN VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Create visualizations
    create_demographic_disparities_chart(df)
    create_intersectional_risk_chart(df)
    create_school_type_chart(df)
    create_geographic_equity_chart(df)
    create_health_coverage_mismatch_chart(df)
    create_summary_infographic(df)
    
    print("\n" + "=" * 70)
    print("VISUALIZATIONS COMPLETE")
    print("=" * 70)
    print(f"\nNew outputs in {FIGURES_DIR}:")
    new_files = [
        'equity_demographic_disparities.png',
        'equity_intersectional_risk.png', 
        'equity_by_school_type.png',
        'equity_geographic_gaps.png',
        'health_coverage_mismatch.png',
        'equity_summary_infographic.png'
    ]
    for f in new_files:
        if (FIGURES_DIR / f).exists():
            print(f"  📊 {f}")


if __name__ == "__main__":
    main()

