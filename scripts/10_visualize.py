#!/usr/bin/env python3
"""
Script 10: Create Visualizations

Input:
    - data/processed/bronx_schools_analysis_ready.csv
    - data/geo/uhf_42_neighborhoods.geojson
    - outputs/tables/neighborhood_summary.csv
    - outputs/tables/non_mshp_schools_priority_ranked.csv

Output:
    - outputs/interactive/mshp_coverage_gap_map.html
    - outputs/figures/absenteeism_comparison.png
    - outputs/figures/priority_tier_distribution.png
    - outputs/figures/neighborhood_coverage_gap.png
    - outputs/figures/coverage_by_school_type.png

This script creates Phase 4 visualizations:
1. Interactive Folium map with schools and asthma choropleth
2. Static comparison charts
3. Priority tier distribution
4. Neighborhood coverage gap chart
"""
import sys
from pathlib import Path

import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import (
    PROCESSED_DIR, GEO_DIR, TABLES_DIR, 
    INTERACTIVE_DIR, FIGURES_DIR, CONFIGS_DIR
)
from mshp_gap.logging_utils import (
    get_logger,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("10_visualize")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_params():
    """Load visualization parameters."""
    params_file = CONFIGS_DIR / "params.yml"
    if params_file.exists():
        with open(params_file) as f:
            return yaml.safe_load(f)
    return {
        "colors": {
            "mshp_covered": "#2E7D32",
            "mshp_not_covered": "#C62828",
            "priority_tiers": {
                "tier_1": "#B71C1C",
                "tier_2": "#FF8C00",
                "tier_3": "#FFD700",
                "tier_4": "#D3D3D3",
            }
        },
        "map": {
            "center_lat": 40.85,
            "center_lon": -73.87,
            "zoom": 12,
        }
    }


def load_data():
    """Load all required data for visualization."""
    log_step_start(logger, "load_data")
    
    # Schools data
    schools = pd.read_csv(PROCESSED_DIR / "bronx_schools_analysis_ready.csv")
    logger.info(f"Loaded {len(schools)} schools")
    
    # UHF boundaries
    uhf_file = GEO_DIR / "uhf_42_neighborhoods.geojson"
    uhf = gpd.read_file(uhf_file)
    uhf = uhf[(uhf['GEOCODE'] >= 101) & (uhf['GEOCODE'] <= 107)]
    logger.info(f"Loaded {len(uhf)} Bronx UHF neighborhoods")
    
    # Neighborhood summary
    neighborhood = pd.read_csv(TABLES_DIR / "neighborhood_summary.csv")
    
    # Priority rankings
    priority = pd.read_csv(TABLES_DIR / "non_mshp_schools_priority_ranked.csv")
    
    log_step_end(logger, "load_data")
    return schools, uhf, neighborhood, priority


def create_interactive_map(schools, uhf, neighborhood, params):
    """Create interactive Folium map with schools and asthma choropleth."""
    log_step_start(logger, "create_interactive_map")
    
    map_params = params.get("map", {})
    colors = params.get("colors", {})
    
    # Create base map
    m = folium.Map(
        location=[map_params.get("center_lat", 40.85), map_params.get("center_lon", -73.87)],
        zoom_start=map_params.get("zoom", 12),
        tiles="cartodbpositron"
    )
    
    # Add asthma choropleth
    # Merge asthma data with UHF boundaries
    uhf_with_data = uhf.merge(
        neighborhood[['uhf_code', 'asthma_ed_rate', 'mshp_coverage_pct', 'coverage_gap_score']],
        left_on='GEOCODE',
        right_on='uhf_code',
        how='left'
    )
    
    # Create choropleth layer
    choropleth = folium.Choropleth(
        geo_data=uhf_with_data.to_json(),
        name="Asthma ED Visit Rate",
        data=neighborhood,
        columns=['uhf_code', 'asthma_ed_rate'],
        key_on='feature.properties.GEOCODE',
        fill_color='YlOrRd',
        fill_opacity=0.5,
        line_opacity=0.8,
        legend_name='Asthma ED Visits per 10,000 Children (5-17)',
        highlight=True,
    )
    choropleth.add_to(m)
    
    # Add neighborhood labels with tooltips
    for _, row in uhf_with_data.iterrows():
        centroid = row.geometry.centroid
        popup_text = f"""
        <b>{row['GEONAME']}</b><br>
        Asthma Rate: {row['asthma_ed_rate']:.1f}<br>
        MSHP Coverage: {row['mshp_coverage_pct']:.0f}%<br>
        Gap Score: {row['coverage_gap_score']:.1f}
        """
        folium.Marker(
            [centroid.y, centroid.x],
            popup=folium.Popup(popup_text, max_width=200),
            icon=folium.DivIcon(
                html=f'<div style="font-size: 8pt; color: #333; font-weight: bold; text-shadow: 1px 1px white;">{row["GEONAME"].split(" - ")[0]}</div>',
                icon_anchor=(30, 10)
            )
        ).add_to(m)
    
    # Create feature groups for schools
    mshp_group = folium.FeatureGroup(name="MSHP Covered Schools")
    non_mshp_group = folium.FeatureGroup(name="Non-MSHP Schools")
    tier1_group = folium.FeatureGroup(name="Tier 1 Critical Priority")
    
    # Add school markers
    for _, school in schools.iterrows():
        # Determine color and group
        if school['has_mshp']:
            color = colors.get("mshp_covered", "#2E7D32")
            icon_color = "green"
            group = mshp_group
        else:
            color = colors.get("mshp_not_covered", "#C62828")
            icon_color = "red"
            group = non_mshp_group
        
        # Size based on enrollment (if available)
        enrollment = school.get('enrollment', 400)
        if pd.isna(enrollment):
            enrollment = 400
        radius = max(4, min(12, enrollment / 100))
        
        # Create popup
        absenteeism = school.get('chronic_absenteeism_rate', 'N/A')
        if pd.notna(absenteeism):
            absenteeism = f"{absenteeism:.1f}%"
        
        popup_html = f"""
        <div style="width: 200px;">
            <b>{school['school_name']}</b><br>
            <hr style="margin: 5px 0;">
            <b>DBN:</b> {school['dbn']}<br>
            <b>Type:</b> {school.get('school_type', 'N/A')}<br>
            <b>MSHP:</b> {'✅ Covered' if school['has_mshp'] else '❌ Not Covered'}<br>
            <b>Absenteeism:</b> {absenteeism}<br>
            <b>Enrollment:</b> {int(enrollment) if pd.notna(enrollment) else 'N/A'}<br>
            <b>Neighborhood:</b> {school.get('uhf_name', 'N/A')}<br>
            <b>Asthma Rate:</b> {school.get('asthma_ed_rate', 'N/A')}
        </div>
        """
        
        marker = folium.CircleMarker(
            location=[school['lat'], school['lon']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=school['school_name']
        )
        marker.add_to(group)
    
    # Add Tier 1 priority markers (larger, with different style)
    priority_schools = schools[(schools['has_mshp'] == False)]
    priority_df = pd.read_csv(TABLES_DIR / "non_mshp_schools_priority_ranked.csv")
    tier1_dbns = priority_df[priority_df['priority_tier'] == 'Tier 1 - Critical']['dbn'].tolist()
    
    for _, school in priority_schools.iterrows():
        if school['dbn'] in tier1_dbns:
            popup_html = f"""
            <div style="width: 220px; border-left: 4px solid #B71C1C; padding-left: 8px;">
                <b style="color: #B71C1C;">🎯 TIER 1 CRITICAL</b><br>
                <b>{school['school_name']}</b><br>
                <hr style="margin: 5px 0;">
                <b>Priority Rank:</b> {priority_df[priority_df['dbn'] == school['dbn']]['priority_rank'].values[0]}<br>
                <b>Absenteeism:</b> {school.get('chronic_absenteeism_rate', 'N/A'):.1f}%<br>
                <b>Asthma Rate:</b> {school.get('asthma_ed_rate', 'N/A')}<br>
                <b>Neighborhood:</b> {school.get('uhf_name', 'N/A')}
            </div>
            """
            
            folium.Marker(
                location=[school['lat'], school['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color='darkred', icon='exclamation-triangle', prefix='fa'),
                tooltip=f"🎯 {school['school_name']}"
            ).add_to(tier1_group)
    
    # Add feature groups to map
    mshp_group.add_to(m)
    non_mshp_group.add_to(m)
    tier1_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px;
                background-color: white; padding: 10px;
                border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                z-index: 9999;">
        <h4 style="margin: 0; color: #333;">MSHP Coverage Gap in the Bronx</h4>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">
            Childhood Asthma Burden vs. School Health Program Coverage
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px;
                background-color: white; padding: 10px;
                border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                z-index: 9999; font-size: 12px;">
        <b>School Markers</b><br>
        <i class="fa fa-circle" style="color:#2E7D32"></i> MSHP Covered<br>
        <i class="fa fa-circle" style="color:#C62828"></i> Not Covered<br>
        <i class="fa fa-exclamation-triangle" style="color:#8B0000"></i> Tier 1 Priority<br>
        <hr style="margin: 5px 0;">
        <small>Circle size = enrollment</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_path = INTERACTIVE_DIR / "mshp_coverage_gap_map.html"
    m.save(str(output_path))
    
    log_output_written(logger, output_path)
    log_step_end(logger, "create_interactive_map")
    
    return output_path


def create_absenteeism_comparison(schools, params):
    """Create bar chart comparing absenteeism between MSHP and non-MSHP schools."""
    log_step_start(logger, "create_absenteeism_comparison")
    
    colors = params.get("colors", {})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter to schools with attendance data
    schools_with_data = schools[schools['chronic_absenteeism_rate'].notna()].copy()
    
    # Bar chart - mean comparison
    ax1 = axes[0]
    mshp_mean = schools_with_data[schools_with_data['has_mshp'] == True]['chronic_absenteeism_rate'].mean()
    non_mshp_mean = schools_with_data[schools_with_data['has_mshp'] == False]['chronic_absenteeism_rate'].mean()
    
    bars = ax1.bar(
        ['MSHP Covered', 'Not Covered'],
        [mshp_mean, non_mshp_mean],
        color=[colors.get("mshp_covered", "#2E7D32"), colors.get("mshp_not_covered", "#C62828")],
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels
    for bar, val in zip(bars, [mshp_mean, non_mshp_mean]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_ylabel('Mean Chronic Absenteeism Rate (%)', fontsize=11)
    ax1.set_title('Chronic Absenteeism by MSHP Status', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(mshp_mean, non_mshp_mean) * 1.2)
    
    # Add sample sizes
    mshp_n = len(schools_with_data[schools_with_data['has_mshp'] == True])
    non_mshp_n = len(schools_with_data[schools_with_data['has_mshp'] == False])
    ax1.text(0, -3, f'n={mshp_n}', ha='center', fontsize=10, color='gray')
    ax1.text(1, -3, f'n={non_mshp_n}', ha='center', fontsize=10, color='gray')
    
    # Add note about significance
    ax1.text(0.5, max(mshp_mean, non_mshp_mean) * 1.1, 
             'Difference not statistically significant (p=0.70)',
             ha='center', fontsize=10, style='italic', color='gray',
             transform=ax1.get_xaxis_transform())
    
    # Box plot - distribution comparison
    ax2 = axes[1]
    schools_with_data['MSHP Status'] = schools_with_data['has_mshp'].map({True: 'MSHP Covered', False: 'Not Covered'})
    
    box = ax2.boxplot(
        [schools_with_data[schools_with_data['has_mshp'] == True]['chronic_absenteeism_rate'],
         schools_with_data[schools_with_data['has_mshp'] == False]['chronic_absenteeism_rate']],
        labels=['MSHP Covered', 'Not Covered'],
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 2}
    )
    
    box['boxes'][0].set_facecolor(colors.get("mshp_covered", "#2E7D32"))
    box['boxes'][1].set_facecolor(colors.get("mshp_not_covered", "#C62828"))
    for patch in box['boxes']:
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Chronic Absenteeism Rate (%)', fontsize=11)
    ax2.set_title('Distribution of Absenteeism Rates', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "absenteeism_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log_output_written(logger, output_path)
    log_step_end(logger, "create_absenteeism_comparison")
    
    return output_path


def create_priority_tier_chart(priority, params):
    """Create chart showing priority tier distribution."""
    log_step_start(logger, "create_priority_tier_chart")
    
    tier_colors = params.get("colors", {}).get("priority_tiers", {})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    ax1 = axes[0]
    tier_counts = priority['priority_tier'].value_counts().sort_index()
    
    colors_list = [
        tier_colors.get("tier_1", "#B71C1C"),
        tier_colors.get("tier_2", "#FF8C00"),
        tier_colors.get("tier_3", "#FFD700"),
        tier_colors.get("tier_4", "#D3D3D3"),
    ]
    
    wedges, texts, autotexts = ax1.pie(
        tier_counts.values,
        labels=tier_counts.index,
        colors=colors_list,
        autopct='%1.0f%%',
        startangle=90,
        explode=[0.05, 0, 0, 0],
        shadow=True
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax1.set_title('Priority Tier Distribution\n(286 Non-MSHP Schools)', fontsize=13, fontweight='bold')
    
    # Bar chart by neighborhood
    ax2 = axes[1]
    
    # Create pivot table
    pivot = priority.groupby(['uhf_name', 'priority_tier']).size().unstack(fill_value=0)
    
    # Reorder columns
    tier_order = ['Tier 1 - Critical', 'Tier 2 - High', 'Tier 3 - Moderate', 'Tier 4 - Lower']
    pivot = pivot.reindex(columns=[c for c in tier_order if c in pivot.columns])
    
    # Sort by Tier 1 count
    if 'Tier 1 - Critical' in pivot.columns:
        pivot = pivot.sort_values('Tier 1 - Critical', ascending=True)
    
    pivot.plot(kind='barh', stacked=True, ax=ax2, color=colors_list, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Number of Schools', fontsize=11)
    ax2.set_ylabel('')
    ax2.set_title('Priority Tiers by Neighborhood', fontsize=13, fontweight='bold')
    ax2.legend(title='Priority Tier', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "priority_tier_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log_output_written(logger, output_path)
    log_step_end(logger, "create_priority_tier_chart")
    
    return output_path


def create_neighborhood_gap_chart(neighborhood, params):
    """Create chart showing coverage gap by neighborhood."""
    log_step_start(logger, "create_neighborhood_gap_chart")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by coverage gap score
    neighborhood_sorted = neighborhood.sort_values('coverage_gap_score', ascending=True)
    
    # Create horizontal bar chart
    y_pos = range(len(neighborhood_sorted))
    
    # Color bars by gap score (higher = more red)
    colors = plt.cm.YlOrRd(neighborhood_sorted['coverage_gap_score'] / 100)
    
    bars = ax.barh(y_pos, neighborhood_sorted['coverage_gap_score'], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(neighborhood_sorted['uhf_name'], fontsize=10)
    ax.set_xlabel('Coverage Gap Score (Higher = Greater Need)', fontsize=11)
    ax.set_title('MSHP Coverage Gap by Neighborhood\n(High Asthma + Low Coverage = High Gap Score)', 
                 fontsize=13, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, neighborhood_sorted['coverage_gap_score']):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=10, fontweight='bold')
    
    # Add annotations for asthma rate and coverage
    for i, (_, row) in enumerate(neighborhood_sorted.iterrows()):
        ax.text(-5, i, f"Asthma: {row['asthma_ed_rate']:.0f} | MSHP: {row['mshp_coverage_pct']:.0f}%",
                va='center', ha='right', fontsize=8, color='gray')
    
    ax.set_xlim(-50, 100)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "neighborhood_coverage_gap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log_output_written(logger, output_path)
    log_step_end(logger, "create_neighborhood_gap_chart")
    
    return output_path


def create_coverage_by_type_chart(schools, params):
    """Create chart showing MSHP coverage by school type."""
    log_step_start(logger, "create_coverage_by_type_chart")
    
    colors = params.get("colors", {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate coverage by type
    coverage_by_type = schools.groupby('school_type').agg({
        'has_mshp': ['sum', 'count']
    }).reset_index()
    coverage_by_type.columns = ['school_type', 'mshp_count', 'total']
    coverage_by_type['coverage_pct'] = (coverage_by_type['mshp_count'] / coverage_by_type['total'] * 100).round(1)
    coverage_by_type['not_covered'] = coverage_by_type['total'] - coverage_by_type['mshp_count']
    
    # Sort by coverage percentage
    coverage_by_type = coverage_by_type.sort_values('coverage_pct', ascending=True)
    
    y_pos = range(len(coverage_by_type))
    
    # Stacked bar chart
    bars1 = ax.barh(y_pos, coverage_by_type['mshp_count'], 
                    color=colors.get("mshp_covered", "#2E7D32"), 
                    label='MSHP Covered', edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_pos, coverage_by_type['not_covered'], 
                    left=coverage_by_type['mshp_count'],
                    color=colors.get("mshp_not_covered", "#C62828"), 
                    label='Not Covered', edgecolor='black', linewidth=0.5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coverage_by_type['school_type'], fontsize=11)
    ax.set_xlabel('Number of Schools', fontsize=11)
    ax.set_title('MSHP Coverage by School Type', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    
    # Add percentage labels
    for i, (_, row) in enumerate(coverage_by_type.iterrows()):
        ax.text(row['total'] + 2, i, f"{row['coverage_pct']:.0f}% covered", 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "coverage_by_school_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log_output_written(logger, output_path)
    log_step_end(logger, "create_coverage_by_type_chart")
    
    return output_path


def main():
    """Main entry point for Phase 4 visualization."""
    logger.info("=" * 60)
    logger.info("PHASE 4: VISUALIZATION")
    logger.info("=" * 60)
    
    # Load data and params
    schools, uhf, neighborhood, priority = load_data()
    params = load_params()
    
    # Create visualizations
    logger.info("\n--- Creating Interactive Map ---")
    map_path = create_interactive_map(schools, uhf, neighborhood, params)
    print(f"✅ Interactive map: {map_path}")
    
    logger.info("\n--- Creating Absenteeism Comparison Chart ---")
    abs_path = create_absenteeism_comparison(schools, params)
    print(f"✅ Absenteeism comparison: {abs_path}")
    
    logger.info("\n--- Creating Priority Tier Chart ---")
    tier_path = create_priority_tier_chart(priority, params)
    print(f"✅ Priority tier distribution: {tier_path}")
    
    logger.info("\n--- Creating Neighborhood Gap Chart ---")
    gap_path = create_neighborhood_gap_chart(neighborhood, params)
    print(f"✅ Neighborhood coverage gap: {gap_path}")
    
    logger.info("\n--- Creating Coverage by Type Chart ---")
    type_path = create_coverage_by_type_chart(schools, params)
    print(f"✅ Coverage by school type: {type_path}")
    
    print("\n" + "=" * 60)
    print("PHASE 4: VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  📍 Interactive Map: {map_path}")
    print(f"  📊 Charts saved to: {FIGURES_DIR}")
    print("\nTo view the interactive map, open the HTML file in a browser.")


if __name__ == "__main__":
    main()

