#!/usr/bin/env python3
"""
Script 12: Enhanced Visualizations

Creates visualizations for the enhanced analysis including:
1. SVI-Asthma-Coverage heatmap
2. Regression coefficient plot
3. Accessibility map
4. Enhanced priority map with all factors
"""
import sys
from pathlib import Path

import folium
from folium import plugins
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import PROCESSED_DIR, GEO_DIR, TABLES_DIR, FIGURES_DIR, INTERACTIVE_DIR

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def load_enhanced_data():
    """Load all enhanced analysis data."""
    schools = pd.read_csv(PROCESSED_DIR / "bronx_schools_enhanced.csv")
    accessibility = pd.read_csv(TABLES_DIR / "accessibility_analysis.csv")
    enhanced_priority = pd.read_csv(TABLES_DIR / "enhanced_priority_ranking.csv")
    regression = pd.read_csv(TABLES_DIR / "regression_results.csv")
    uhf = gpd.read_file(GEO_DIR / "uhf_42_neighborhoods.geojson")
    uhf = uhf[(uhf['GEOCODE'] >= 101) & (uhf['GEOCODE'] <= 107)]
    return schools, accessibility, enhanced_priority, regression, uhf


def create_svi_asthma_heatmap(schools):
    """Create heatmap showing SVI vs Asthma by neighborhood."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Aggregate by neighborhood
    neighborhood_data = schools.groupby('uhf_name').agg({
        'svi_overall': 'first',
        'svi_socioeconomic': 'first',
        'svi_household_disability': 'first',
        'svi_minority_language': 'first',
        'svi_housing_transport': 'first',
        'asthma_ed_rate': 'first',
        'has_mshp': 'mean',  # coverage rate
        'chronic_absenteeism_rate': 'mean'
    }).reset_index()
    
    # Rename for display
    neighborhood_data.columns = ['Neighborhood', 'SVI Overall', 'SVI Socioeconomic', 
                                  'SVI Household', 'SVI Minority', 'SVI Housing',
                                  'Asthma Rate', 'MSHP Coverage', 'Avg Absenteeism']
    
    # Heatmap 1: Neighborhood vulnerability profile
    ax1 = axes[0]
    heatmap_data = neighborhood_data.set_index('Neighborhood')[
        ['SVI Socioeconomic', 'SVI Household', 'SVI Minority', 'SVI Housing']
    ]
    heatmap_data = heatmap_data.sort_values('SVI Socioeconomic', ascending=False)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1,
                cbar_kws={'label': 'Vulnerability Score (0-1)'})
    ax1.set_title('Social Vulnerability by Neighborhood', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Scatter: SVI vs Coverage Gap
    ax2 = axes[1]
    colors = plt.cm.YlOrRd(neighborhood_data['Asthma Rate'] / neighborhood_data['Asthma Rate'].max())
    
    scatter = ax2.scatter(
        neighborhood_data['SVI Overall'],
        neighborhood_data['MSHP Coverage'] * 100,
        s=neighborhood_data['Asthma Rate'] * 2,
        c=neighborhood_data['Asthma Rate'],
        cmap='YlOrRd',
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add labels
    for _, row in neighborhood_data.iterrows():
        short_name = row['Neighborhood'].split(' - ')[0]
        ax2.annotate(short_name, (row['SVI Overall'], row['MSHP Coverage'] * 100),
                    fontsize=8, ha='center', va='bottom')
    
    ax2.set_xlabel('Social Vulnerability Index (0-1)', fontsize=11)
    ax2.set_ylabel('MSHP Coverage (%)', fontsize=11)
    ax2.set_title('Vulnerability vs MSHP Coverage\n(Bubble size = Asthma Rate)', 
                  fontsize=12, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax2, label='Asthma ED Rate per 10K')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "svi_asthma_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ SVI-Asthma heatmap: {output_path}")
    return output_path


def create_regression_plot(regression):
    """Create forest plot of regression coefficients."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Extract MSHP coefficients and p-values
    models = regression['model'].tolist()
    coefs = regression['mshp_coefficient'].tolist()
    pvalues = regression['mshp_pvalue'].tolist()
    
    # Create forest plot
    y_positions = range(len(models))
    colors = ['#27AE60' if p < 0.05 else '#95A5A6' for p in pvalues]
    
    ax.barh(y_positions, coefs, color=colors, edgecolor='black', height=0.6)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add labels
    ax.set_yticks(y_positions)
    labels = [
        'Simple (MSHP only)',
        'With Asthma Control',
        'Full Model (all controls)',
        'With Interaction Term'
    ]
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add p-value annotations
    for i, (coef, pval) in enumerate(zip(coefs, pvalues)):
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        ax.text(coef + 0.3, i, f'β={coef:.2f}, p={pval:.3f}{sig}', 
                va='center', fontsize=9)
    
    ax.set_xlabel('MSHP Coefficient (Effect on Chronic Absenteeism %)', fontsize=11)
    ax.set_title('Effect of MSHP Coverage on Chronic Absenteeism\n(Controlling for Confounders)', 
                 fontsize=13, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27AE60', edgecolor='black', label='Significant (p<0.05)'),
        Patch(facecolor='#95A5A6', edgecolor='black', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add interpretation note
    ax.text(0.02, 0.02, 
            "Note: After controlling for asthma burden, SVI, enrollment, and school type,\n"
            "MSHP coverage shows no significant effect on absenteeism.\n"
            "This suggests MSHP may be targeting high-need schools (selection effect).",
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "regression_coefficients.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Regression plot: {output_path}")
    return output_path


def create_accessibility_chart(accessibility):
    """Create accessibility analysis chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of distances
    ax1 = axes[0]
    ax1.hist(accessibility['distance_to_nearest_mshp_miles'], bins=30, 
             color='#3498DB', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=2, label='0.5 mile threshold')
    ax1.axvline(x=1.0, color='#C0392B', linestyle='--', linewidth=2, label='1 mile threshold')
    ax1.axvline(x=accessibility['distance_to_nearest_mshp_miles'].mean(), 
                color='#27AE60', linestyle='-', linewidth=2, label='Mean')
    
    ax1.set_xlabel('Distance to Nearest MSHP School (miles)', fontsize=11)
    ax1.set_ylabel('Number of Schools', fontsize=11)
    ax1.set_title('Distribution of Distance to Nearest MSHP School', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Bar chart by neighborhood
    ax2 = axes[1]
    
    # Aggregate by neighborhood
    neigh_access = accessibility.groupby('uhf_name').agg({
        'distance_to_nearest_mshp_miles': 'mean',
        'dbn': 'count'
    }).reset_index()
    neigh_access.columns = ['Neighborhood', 'Avg Distance', 'Schools']
    neigh_access = neigh_access.sort_values('Avg Distance', ascending=True)
    
    colors = ['#E74C3C' if d > 0.5 else '#27AE60' for d in neigh_access['Avg Distance']]
    
    bars = ax2.barh(neigh_access['Neighborhood'], neigh_access['Avg Distance'], 
                    color=colors, edgecolor='black')
    
    ax2.axvline(x=0.5, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Average Distance to Nearest MSHP (miles)', fontsize=11)
    ax2.set_title('Accessibility by Neighborhood', fontsize=12, fontweight='bold')
    
    # Add counts
    for bar, count in zip(bars, neigh_access['Schools']):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'n={count}', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "accessibility_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Accessibility chart: {output_path}")
    return output_path


def create_enhanced_priority_map(schools, enhanced_priority, uhf, accessibility):
    """Create interactive map with all enhanced features."""
    
    # Merge accessibility into priority
    enhanced_priority = enhanced_priority.merge(
        accessibility[['dbn', 'nearest_mshp_dbn', 'nearest_mshp_name']],
        on='dbn', how='left', suffixes=('', '_acc')
    )
    
    # Create map
    m = folium.Map(
        location=[40.855, -73.88],
        zoom_start=12,
        tiles='cartodbdark_matter'
    )
    
    # Add neighborhood choropleth by SVI
    svi_by_uhf = schools.groupby('uhf_code')['svi_overall'].first().to_dict()
    uhf['svi_overall'] = uhf['GEOCODE'].map(svi_by_uhf)
    
    folium.Choropleth(
        geo_data=uhf.to_json(),
        name='Social Vulnerability Index',
        data=uhf,
        columns=['GEOCODE', 'svi_overall'],
        key_on='feature.properties.GEOCODE',
        fill_color='PuRd',
        fill_opacity=0.5,
        line_opacity=0.8,
        legend_name='Social Vulnerability Index',
        highlight=True,
    ).add_to(m)
    
    # Add MSHP schools (for reference)
    mshp_group = folium.FeatureGroup(name='MSHP Schools (Reference)')
    mshp_schools = schools[schools['has_mshp'] == True]
    
    for _, school in mshp_schools.iterrows():
        folium.CircleMarker(
            location=[school['lat'], school['lon']],
            radius=6,
            color='#27AE60',
            fill=True,
            fillColor='#27AE60',
            fillOpacity=0.8,
            popup=school['school_name'],
            tooltip=f"MSHP: {school['school_name']}"
        ).add_to(mshp_group)
    mshp_group.add_to(m)
    
    # Add priority schools by tier
    tier_groups = {
        'Tier 1 - Critical (Top 10%)': folium.FeatureGroup(name='🎯 Tier 1 Critical (Top 10%)'),
        'Tier 2 - High (10-25%)': folium.FeatureGroup(name='🟠 Tier 2 High (10-25%)'),
        'Tier 3 - Moderate (25-50%)': folium.FeatureGroup(name='🟡 Tier 3 Moderate'),
        'Tier 4 - Lower (Bottom 50%)': folium.FeatureGroup(name='⚪ Tier 4 Lower'),
    }
    
    tier_colors = {
        'Tier 1 - Critical (Top 10%)': '#B71C1C',
        'Tier 2 - High (10-25%)': '#FF8C00',
        'Tier 3 - Moderate (25-50%)': '#FFD700',
        'Tier 4 - Lower (Bottom 50%)': '#808080',
    }
    
    for _, school in enhanced_priority.iterrows():
        tier = school['enhanced_priority_tier']
        color = tier_colors.get(tier, '#808080')
        
        # Size by priority score
        radius = 4 + (school['enhanced_priority_score'] / 100) * 10
        
        popup_html = f"""
        <div style="font-family: Arial; width: 280px; background: #1a1a1a; color: white; padding: 12px; border-radius: 8px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">{school['school_name']}</h4>
            <div style="font-size: 11px; color: #bbb; margin-bottom: 8px;">Rank #{int(school['enhanced_priority_rank'])} | {tier}</div>
            <table style="width: 100%; font-size: 11px;">
                <tr><td style="color: #888;">Priority Score</td><td style="text-align: right; font-weight: bold;">{school['enhanced_priority_score']:.1f}</td></tr>
                <tr><td style="color: #888;">SVI Score</td><td style="text-align: right;">{school.get('svi_overall', 0):.2f}</td></tr>
                <tr><td style="color: #888;">Asthma Rate</td><td style="text-align: right;">{school.get('asthma_ed_rate', 0):.0f} per 10K</td></tr>
                <tr><td style="color: #888;">Absenteeism</td><td style="text-align: right;">{school.get('chronic_absenteeism_rate', 0):.1f}%</td></tr>
                <tr><td style="color: #888;">Distance to MSHP</td><td style="text-align: right;">{school.get('distance_to_nearest_mshp_miles', 0):.2f} mi</td></tr>
                <tr><td style="color: #888;">Nearest MSHP</td><td style="text-align: right; font-size: 10px;">{str(school.get('nearest_mshp_name', 'N/A'))[:25]}</td></tr>
            </table>
        </div>
        """
        
        marker = folium.CircleMarker(
            location=[school['lat'], school['lon']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{int(school['enhanced_priority_rank'])} {school['school_name']}"
        )
        
        group = tier_groups.get(tier)
        if group:
            marker.add_to(group)
    
    for group in tier_groups.values():
        group.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; top: 15px; left: 60px; z-index: 9999;
                background: rgba(0,0,0,0.85); padding: 20px; border-radius: 12px;
                font-family: Arial; max-width: 350px; color: white;">
        <h3 style="margin: 0 0 8px 0; color: #ECF0F1;">Enhanced Priority Map</h3>
        <p style="margin: 0 0 10px 0; font-size: 12px; color: #BDC3C7;">
            Schools ranked by composite equity score:<br>
            SVI + Asthma + Absenteeism + Isolation
        </p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 5px; font-size: 11px;">
            <div><span style="color: #B71C1C;">●</span> Tier 1: 28 schools</div>
            <div><span style="color: #FF8C00;">●</span> Tier 2: 43 schools</div>
            <div><span style="color: #FFD700;">●</span> Tier 3: 72 schools</div>
            <div><span style="color: #808080;">●</span> Tier 4: 143 schools</div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add plugins
    plugins.Fullscreen().add_to(m)
    plugins.MiniMap(toggle_display=True).add_to(m)
    
    # Save
    output_path = INTERACTIVE_DIR / "enhanced_priority_map.html"
    m.save(str(output_path))
    print(f"✅ Enhanced priority map: {output_path}")
    return output_path


def create_composite_summary_chart(enhanced_priority):
    """Create summary chart of enhanced priority factors."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Score distribution by tier
    ax1 = axes[0, 0]
    tier_order = ['Tier 1 - Critical (Top 10%)', 'Tier 2 - High (10-25%)', 
                  'Tier 3 - Moderate (25-50%)', 'Tier 4 - Lower (Bottom 50%)']
    colors = ['#B71C1C', '#FF8C00', '#FFD700', '#808080']
    
    for tier, color in zip(tier_order, colors):
        data = enhanced_priority[enhanced_priority['enhanced_priority_tier'] == tier]['enhanced_priority_score']
        ax1.hist(data, bins=15, alpha=0.7, label=tier.split(' (')[0], color=color, edgecolor='black')
    
    ax1.set_xlabel('Enhanced Priority Score', fontsize=11)
    ax1.set_ylabel('Number of Schools', fontsize=11)
    ax1.set_title('Score Distribution by Tier', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # Top right: Component contribution
    ax2 = axes[0, 1]
    components = ['Asthma\n(25%)', 'SVI\n(25%)', 'Absenteeism\n(20%)', 
                  'Isolation\n(15%)', 'Enrollment\n(15%)']
    
    # Get averages for top 10 schools
    top10 = enhanced_priority.head(10)
    top10_scores = [
        top10['asthma_score'].mean(),
        top10['svi_score'].mean(),
        top10['absenteeism_score'].mean(),
        top10['isolation_score'].mean(),
        top10['enrollment_score'].mean()
    ]
    
    all_scores = [
        enhanced_priority['asthma_score'].mean(),
        enhanced_priority['svi_score'].mean(),
        enhanced_priority['absenteeism_score'].mean(),
        enhanced_priority['isolation_score'].mean(),
        enhanced_priority['enrollment_score'].mean()
    ]
    
    x = np.arange(len(components))
    width = 0.35
    
    ax2.bar(x - width/2, top10_scores, width, label='Top 10 Priority', color='#B71C1C', edgecolor='black')
    ax2.bar(x + width/2, all_scores, width, label='All Non-MSHP', color='#95A5A6', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.set_ylabel('Average Component Score', fontsize=11)
    ax2.set_title('What Makes Top Priority Schools Different?', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # Bottom left: Correlation matrix
    ax3 = axes[1, 0]
    corr_cols = ['asthma_score', 'svi_score', 'absenteeism_score', 
                 'isolation_score', 'enrollment_score', 'enhanced_priority_score']
    corr_data = enhanced_priority[corr_cols].corr()
    corr_data.columns = ['Asthma', 'SVI', 'Absent.', 'Isolation', 'Enroll.', 'Priority']
    corr_data.index = ['Asthma', 'SVI', 'Absent.', 'Isolation', 'Enroll.', 'Priority']
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax3,
                vmin=-1, vmax=1)
    ax3.set_title('Factor Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Bottom right: Top 15 schools
    ax4 = axes[1, 1]
    top15 = enhanced_priority.head(15)
    
    tier_colors_local = {
        'Tier 1 - Critical (Top 10%)': '#B71C1C',
        'Tier 2 - High (10-25%)': '#FF8C00',
        'Tier 3 - Moderate (25-50%)': '#FFD700',
        'Tier 4 - Lower (Bottom 50%)': '#808080',
    }
    
    y_pos = range(len(top15))
    colors = [tier_colors_local.get(t, '#808080') for t in top15['enhanced_priority_tier']]
    
    bars = ax4.barh(y_pos, top15['enhanced_priority_score'], color=colors, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"{i+1}. {n[:30]}..." if len(n) > 30 else f"{i+1}. {n}" 
                         for i, n in enumerate(top15['school_name'])], fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel('Enhanced Priority Score', fontsize=11)
    ax4.set_title('Top 15 Priority Schools', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "enhanced_priority_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Enhanced priority summary: {output_path}")
    return output_path


def main():
    """Create all enhanced visualizations."""
    print("=" * 60)
    print("ENHANCED VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    schools, accessibility, enhanced_priority, regression, uhf = load_enhanced_data()
    print(f"Loaded {len(schools)} schools, {len(enhanced_priority)} priority rankings")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_svi_asthma_heatmap(schools)
    create_regression_plot(regression)
    create_accessibility_chart(accessibility)
    create_enhanced_priority_map(schools, enhanced_priority, uhf, accessibility)
    create_composite_summary_chart(enhanced_priority)
    
    print("\n" + "=" * 60)
    print("ENHANCED VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"\n📊 Static charts: {FIGURES_DIR}")
    print(f"📍 Interactive map: {INTERACTIVE_DIR / 'enhanced_priority_map.html'}")


if __name__ == "__main__":
    main()

