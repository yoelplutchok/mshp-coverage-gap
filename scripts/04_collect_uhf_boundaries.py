#!/usr/bin/env python3
"""
Script 04b: Collect UHF 42 Neighborhood Boundaries

Input:  None (fetches from NYC Open Data API)
Output: data/geo/uhf_42_neighborhoods.geojson

Downloads UHF 42 neighborhood boundaries from NYC Open Data.
"""
import sys
from datetime import datetime

import geopandas as gpd
import requests

from mshp_gap.paths import GEO_DIR, CONFIGS_DIR
from mshp_gap.io_utils import write_metadata_sidecar, atomic_write_json
from mshp_gap.logging_utils import (
    get_logger,
    get_run_id,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("04_collect_uhf_boundaries")


def download_uhf_boundaries() -> gpd.GeoDataFrame:
    """
    Download UHF 42 neighborhood boundaries from NYC Open Data.
    
    Dataset: UHF 42 Neighborhoods
    """
    # UHF 42 boundaries dataset on NYC Open Data
    # This is the GeoJSON endpoint
    dataset_id = "cwiz-gcty"  # UHF 42 Neighborhoods
    api_url = f"https://data.cityofnewyork.us/resource/{dataset_id}.geojson"
    
    logger.info(f"Fetching UHF boundaries from {api_url}")
    
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    
    # Load GeoJSON directly into GeoDataFrame
    gdf = gpd.read_file(api_url)
    
    logger.info(f"Downloaded {len(gdf)} UHF neighborhoods")
    
    return gdf


def filter_bronx_uhf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to only Bronx UHF neighborhoods."""
    # Bronx UHF codes are typically 101-107
    # The column name might vary - let's check what we have
    logger.info(f"Available columns: {gdf.columns.tolist()}")
    
    # Try to identify the UHF code/name column
    bronx_uhf_names = [
        "kingsbridge", "riverdale", "northeast", "fordham", "bronx park",
        "pelham", "throgs neck", "crotona", "tremont", "high bridge",
        "morrisania", "hunts point", "mott haven"
    ]
    
    # Look for a name column
    name_col = None
    for col in gdf.columns:
        if "name" in col.lower() or "uhf" in col.lower():
            name_col = col
            break
    
    if name_col:
        logger.info(f"Using column '{name_col}' to identify neighborhoods")
        # Filter for Bronx neighborhoods
        bronx_mask = gdf[name_col].str.lower().str.contains(
            "|".join(bronx_uhf_names), na=False
        )
        bronx_gdf = gdf[bronx_mask].copy()
        
        if len(bronx_gdf) == 0:
            logger.warning("No Bronx neighborhoods found with name filter, returning all")
            return gdf
        
        logger.info(f"Filtered to {len(bronx_gdf)} Bronx UHF neighborhoods")
        return bronx_gdf
    
    # If we can't filter, return all and we'll filter later
    logger.warning("Could not identify UHF name column, returning all neighborhoods")
    return gdf


def main():
    """Main entry point."""
    log_step_start(logger, "download_uhf_boundaries")
    
    try:
        # Download boundaries
        gdf = download_uhf_boundaries()
        
        # We'll save all UHFs (we need them for the spatial join)
        # but also identify which are Bronx
        bronx_gdf = filter_bronx_uhf(gdf)
        
        # Ensure CRS is WGS84
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Save full UHF file
        output_path = GEO_DIR / "uhf_42_neighborhoods.geojson"
        gdf.to_file(output_path, driver="GeoJSON")
        log_output_written(logger, output_path, row_count=len(gdf))
        
        # Write metadata sidecar
        write_metadata_sidecar(
            data_path=output_path,
            script_name="04_collect_uhf_boundaries.py",
            run_id=get_run_id(),
            description="UHF 42 neighborhood boundaries from NYC Open Data",
            inputs=["NYC Open Data API: dataset cwiz-gcty"],
            row_count=len(gdf),
            columns=list(gdf.columns),
            bronx_neighborhoods=len(bronx_gdf),
        )
        
        log_step_end(logger, "download_uhf_boundaries", neighborhoods=len(gdf))
        
        # Print summary
        print(f"\n✅ Successfully downloaded {len(gdf)} UHF neighborhoods")
        print(f"   Bronx neighborhoods identified: {len(bronx_gdf)}")
        print(f"   Output: {output_path}")
        
        # List Bronx neighborhoods
        if len(bronx_gdf) > 0:
            name_col = [c for c in bronx_gdf.columns if "name" in c.lower()][0] if any("name" in c.lower() for c in bronx_gdf.columns) else bronx_gdf.columns[0]
            print(f"\n   Bronx UHF Neighborhoods:")
            for name in bronx_gdf[name_col].values:
                print(f"     - {name}")
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

