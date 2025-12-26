#!/usr/bin/env python3
"""
Script 10b: Create Enhanced Visualizations

This version creates a more polished, visually striking interactive map.
"""
import sys
from pathlib import Path

import folium
from folium import plugins
import geopandas as gpd
import branca.colormap as cm
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import (
    PROCESSED_DIR, GEO_DIR, TABLES_DIR, 
    INTERACTIVE_DIR, CONFIGS_DIR
)
from mshp_gap.logging_utils import get_logger

logger = get_logger("10_visualize_enhanced")


def load_data():
    """Load all required data."""
    schools = pd.read_csv(PROCESSED_DIR / "bronx_schools_analysis_ready.csv")
    uhf = gpd.read_file(GEO_DIR / "uhf_42_neighborhoods.geojson")
    uhf = uhf[(uhf['GEOCODE'] >= 101) & (uhf['GEOCODE'] <= 107)]
    neighborhood = pd.read_csv(TABLES_DIR / "neighborhood_summary.csv")
    priority = pd.read_csv(TABLES_DIR / "non_mshp_schools_priority_ranked.csv")
    return schools, uhf, neighborhood, priority


def create_enhanced_map(schools, uhf, neighborhood, priority):
    """Create a visually enhanced interactive map."""
    
    # Merge data
    uhf_with_data = uhf.merge(
        neighborhood[['uhf_code', 'asthma_ed_rate', 'mshp_coverage_pct', 'coverage_gap_score']],
        left_on='GEOCODE', right_on='uhf_code', how='left'
    )
    
    # Create base map with dark elegant tiles
    m = folium.Map(
        location=[40.855, -73.88],
        zoom_start=12,
        tiles=None,  # We'll add custom tiles
        prefer_canvas=True
    )
    
    # Add multiple tile layer options
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Dark Mode',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='Light Mode',
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        control=True
    ).add_to(m)
    
    # Custom color scheme - more vibrant
    asthma_colormap = cm.LinearColormap(
        colors=['#2C3E50', '#E74C3C', '#C0392B'],  # Dark blue to vibrant red
        vmin=neighborhood['asthma_ed_rate'].min(),
        vmax=neighborhood['asthma_ed_rate'].max(),
        caption='Asthma ED Visits per 10,000 Children (5-17)'
    )
    
    # Add choropleth with custom styling
    def style_function(feature):
        asthma_rate = feature['properties'].get('asthma_ed_rate', 100)
        return {
            'fillColor': asthma_colormap(asthma_rate) if asthma_rate else '#555',
            'color': '#ECF0F1',  # Light border
            'weight': 2,
            'fillOpacity': 0.65,
            'dashArray': '3'
        }
    
    def highlight_function(feature):
        return {
            'fillColor': '#F39C12',  # Gold highlight
            'color': '#FFFFFF',
            'weight': 4,
            'fillOpacity': 0.8
        }
    
    # Create neighborhood layer with custom tooltips
    neighborhoods_layer = folium.GeoJson(
        uhf_with_data.to_json(),
        name='Neighborhoods (Asthma Burden)',
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['GEONAME', 'asthma_ed_rate', 'mshp_coverage_pct'],
            aliases=['<b>Neighborhood</b>', '<b>Asthma Rate</b>', '<b>MSHP Coverage</b>'],
            localize=True,
            sticky=True,
            style='''
                background-color: rgba(0, 0, 0, 0.85);
                color: #ECF0F1;
                font-family: 'Segoe UI', Tahoma, sans-serif;
                font-size: 13px;
                padding: 12px;
                border-radius: 8px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            '''
        )
    )
    neighborhoods_layer.add_to(m)
    
    # Add colormap to map
    asthma_colormap.add_to(m)
    
    # Create feature groups
    mshp_group = folium.FeatureGroup(name='✅ MSHP Covered Schools', show=True)
    non_mshp_group = folium.FeatureGroup(name='❌ Non-MSHP Schools', show=True)
    tier1_group = folium.FeatureGroup(name='🎯 Tier 1 Critical Priority', show=True)
    
    # Get tier 1 DBNs
    tier1_dbns = set(priority[priority['priority_tier'] == 'Tier 1 - Critical']['dbn'].tolist())
    
    # Enhanced school markers
    for _, school in schools.iterrows():
        is_mshp = school['has_mshp']
        is_tier1 = school['dbn'] in tier1_dbns
        
        # Get enrollment for sizing
        enrollment = school.get('enrollment', 400)
        if pd.isna(enrollment):
            enrollment = 400
        radius = max(5, min(14, enrollment / 80))
        
        # Styling based on status
        if is_mshp:
            fill_color = '#27AE60'  # Emerald green
            border_color = '#1E8449'
            opacity = 0.85
            group = mshp_group
            icon_text = '✓'
        else:
            fill_color = '#E74C3C'  # Vibrant red
            border_color = '#C0392B'
            opacity = 0.75
            group = non_mshp_group
            icon_text = '✗'
        
        # Format absenteeism
        absenteeism = school.get('chronic_absenteeism_rate')
        if pd.notna(absenteeism):
            abs_display = f"{absenteeism:.1f}%"
            # Color code absenteeism
            if absenteeism > 45:
                abs_color = '#E74C3C'
            elif absenteeism > 35:
                abs_color = '#F39C12'
            else:
                abs_color = '#27AE60'
        else:
            abs_display = 'N/A'
            abs_color = '#95A5A6'
        
        # Get priority rank if non-MSHP
        priority_info = ""
        if not is_mshp:
            match = priority[priority['dbn'] == school['dbn']]
            if len(match) > 0:
                rank = match['priority_rank'].values[0]
                tier = match['priority_tier'].values[0]
                tier_color = {
                    'Tier 1 - Critical': '#E74C3C',
                    'Tier 2 - High': '#F39C12',
                    'Tier 3 - Moderate': '#F1C40F',
                    'Tier 4 - Lower': '#95A5A6'
                }.get(tier, '#95A5A6')
                priority_info = f"""
                <div style="margin-top: 8px; padding: 8px; background: linear-gradient(135deg, {tier_color}22, {tier_color}44); border-left: 3px solid {tier_color}; border-radius: 4px;">
                    <div style="font-weight: 600; color: {tier_color};">📊 Priority Rank #{rank}</div>
                    <div style="font-size: 11px; color: #BDC3C7;">{tier}</div>
                </div>
                """
        
        # Create beautiful popup
        popup_html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, sans-serif; width: 280px; padding: 0;">
            <div style="background: linear-gradient(135deg, {'#27AE60' if is_mshp else '#E74C3C'}, {'#1E8449' if is_mshp else '#C0392B'}); padding: 12px; border-radius: 8px 8px 0 0; margin: -1px;">
                <div style="font-size: 14px; font-weight: 700; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">
                    {school['school_name']}
                </div>
                <div style="font-size: 11px; color: rgba(255,255,255,0.85); margin-top: 4px;">
                    {'✅ MSHP Covered' if is_mshp else '❌ Not Covered by MSHP'}
                </div>
            </div>
            <div style="padding: 12px; background: #2C3E50; border-radius: 0 0 8px 8px;">
                <table style="width: 100%; font-size: 12px; color: #ECF0F1;">
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">DBN</td>
                        <td style="padding: 4px 0; text-align: right; font-weight: 600;">{school['dbn']}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">Type</td>
                        <td style="padding: 4px 0; text-align: right;">{school.get('school_type', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">Enrollment</td>
                        <td style="padding: 4px 0; text-align: right;">{int(enrollment):,}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">Chronic Absenteeism</td>
                        <td style="padding: 4px 0; text-align: right; font-weight: 700; color: {abs_color};">{abs_display}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">Neighborhood</td>
                        <td style="padding: 4px 0; text-align: right;">{school.get('uhf_name', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 4px 0; color: #95A5A6;">Asthma Rate</td>
                        <td style="padding: 4px 0; text-align: right;">{school.get('asthma_ed_rate', 'N/A'):.0f} per 10K</td>
                    </tr>
                </table>
                {priority_info}
            </div>
        </div>
        """
        
        # Create marker with glow effect for MSHP schools
        if is_mshp:
            # Outer glow
            folium.CircleMarker(
                location=[school['lat'], school['lon']],
                radius=radius + 4,
                color=fill_color,
                fill=True,
                fillColor=fill_color,
                fillOpacity=0.2,
                weight=0,
            ).add_to(group)
        
        # Main marker
        marker = folium.CircleMarker(
            location=[school['lat'], school['lon']],
            radius=radius,
            color=border_color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=opacity,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"<b>{school['school_name']}</b><br>{'✅ MSHP' if is_mshp else '❌ Not Covered'}"
        )
        marker.add_to(group)
        
        # Add Tier 1 markers with special icon
        if is_tier1:
            folium.Marker(
                location=[school['lat'], school['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"🎯 HIGH PRIORITY: {school['school_name']}",
                icon=folium.DivIcon(
                    html=f'''
                    <div style="
                        font-size: 20px;
                        color: #E74C3C;
                        text-shadow: 0 0 8px #E74C3C, 0 0 16px #E74C3C;
                        animation: pulse 1.5s infinite;
                    ">🎯</div>
                    <style>
                        @keyframes pulse {{
                            0%, 100% {{ transform: scale(1); opacity: 1; }}
                            50% {{ transform: scale(1.2); opacity: 0.8; }}
                        }}
                    </style>
                    ''',
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                )
            ).add_to(tier1_group)
    
    # Add feature groups
    mshp_group.add_to(m)
    non_mshp_group.add_to(m)
    tier1_group.add_to(m)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add minimap
    minimap = plugins.MiniMap(
        tile_layer=folium.TileLayer(
            tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            attr='CARTO'
        ),
        toggle_display=True,
        minimized=False,
        position='bottomleft'
    )
    minimap.add_to(m)
    
    # Add custom title and stats panel
    title_html = '''
    <div style="
        position: fixed;
        top: 15px;
        left: 60px;
        z-index: 9999;
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        padding: 20px 25px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Segoe UI', Tahoma, sans-serif;
        max-width: 380px;
    ">
        <h2 style="
            margin: 0 0 8px 0;
            color: #ECF0F1;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
        ">
            MSHP Coverage Gap in the Bronx
        </h2>
        <p style="
            margin: 0 0 12px 0;
            color: #BDC3C7;
            font-size: 13px;
            line-height: 1.4;
        ">
            Mapping childhood asthma burden against<br>
            Montefiore School Health Program coverage
        </p>
        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 12px;
        ">
            <div style="text-align: center; padding: 8px; background: rgba(39, 174, 96, 0.2); border-radius: 8px;">
                <div style="font-size: 22px; font-weight: 700; color: #27AE60;">84</div>
                <div style="font-size: 10px; color: #95A5A6; text-transform: uppercase;">Covered</div>
            </div>
            <div style="text-align: center; padding: 8px; background: rgba(231, 76, 60, 0.2); border-radius: 8px;">
                <div style="font-size: 22px; font-weight: 700; color: #E74C3C;">286</div>
                <div style="font-size: 10px; color: #95A5A6; text-transform: uppercase;">Not Covered</div>
            </div>
            <div style="text-align: center; padding: 8px; background: rgba(243, 156, 18, 0.2); border-radius: 8px;">
                <div style="font-size: 22px; font-weight: 700; color: #F39C12;">58</div>
                <div style="font-size: 10px; color: #95A5A6; text-transform: uppercase;">Critical</div>
            </div>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add legend
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 30px;
        right: 15px;
        z-index: 9999;
        background: linear-gradient(135deg, rgba(44, 62, 80, 0.95), rgba(52, 73, 94, 0.95));
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Segoe UI', Tahoma, sans-serif;
    ">
        <div style="font-weight: 600; color: #ECF0F1; margin-bottom: 10px; font-size: 13px;">School Markers</div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 16px; height: 16px; background: #27AE60; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 8px #27AE60;"></div>
            <span style="color: #ECF0F1; font-size: 12px;">MSHP Covered</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="width: 16px; height: 16px; background: #E74C3C; border-radius: 50%; margin-right: 10px;"></div>
            <span style="color: #ECF0F1; font-size: 12px;">Not Covered</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <div style="margin-right: 10px; font-size: 16px;">🎯</div>
            <span style="color: #ECF0F1; font-size: 12px;">Tier 1 Critical</span>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
            <div style="color: #95A5A6; font-size: 11px;">Circle size = enrollment</div>
        </div>
        <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
            <a href="https://github.com/yoelplutchok/mshp-coverage-gap" target="_blank" 
               style="display: flex; align-items: center; color: #58a6ff; text-decoration: none; font-size: 11px;">
                <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" style="margin-right: 6px;">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                View on GitHub
            </a>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add CSS animations
    css = '''
    <style>
        .leaflet-tooltip {
            background-color: rgba(44, 62, 80, 0.95) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: #ECF0F1 !important;
            font-family: 'Segoe UI', Tahoma, sans-serif !important;
            padding: 8px 12px !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        }
        .leaflet-tooltip-left:before,
        .leaflet-tooltip-right:before {
            border-left-color: rgba(44, 62, 80, 0.95) !important;
            border-right-color: rgba(44, 62, 80, 0.95) !important;
        }
        .leaflet-popup-content-wrapper {
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        .leaflet-popup-content {
            margin: 0 !important;
        }
        .leaflet-popup-tip {
            background: #2C3E50 !important;
        }
        .leaflet-control-layers {
            background: rgba(44, 62, 80, 0.95) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #ECF0F1 !important;
        }
        .leaflet-control-layers-base label,
        .leaflet-control-layers-overlays label {
            color: #ECF0F1 !important;
        }
    </style>
    '''
    m.get_root().html.add_child(folium.Element(css))
    
    # Save map
    output_path = INTERACTIVE_DIR / "mshp_coverage_gap_map_enhanced.html"
    m.save(str(output_path))
    
    print(f"✅ Enhanced map saved: {output_path}")
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Creating Enhanced Interactive Map")
    print("=" * 60)
    
    schools, uhf, neighborhood, priority = load_data()
    output_path = create_enhanced_map(schools, uhf, neighborhood, priority)
    
    print("\n🎨 Enhancement Features:")
    print("   • Dark/Light/Satellite tile options")
    print("   • Gradient color scheme")
    print("   • Glowing markers for MSHP schools")
    print("   • Pulsing 🎯 icons for Tier 1 priority")
    print("   • Beautiful styled popups")
    print("   • Stats panel with key metrics")
    print("   • Minimap for context")
    print("   • Fullscreen button")
    print(f"\n📍 Open: {output_path}")


if __name__ == "__main__":
    main()

