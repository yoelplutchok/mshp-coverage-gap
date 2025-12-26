#!/usr/bin/env python3
"""
Script 01: Collect DOE School Locations

Input:  None (fetches from NYC Open Data API)
Output: data/raw/doe_school_locations_YYYY-MM-DD.csv

Downloads all Bronx public school locations with coordinates
from the NYC Open Data Socrata API.
"""
import sys
from datetime import datetime

import pandas as pd
import requests
import yaml

from mshp_gap.paths import RAW_DIR, CONFIGS_DIR
from mshp_gap.io_utils import atomic_write_csv, write_metadata_sidecar, update_manifest
from mshp_gap.logging_utils import (
    get_logger,
    get_run_id,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("01_collect_doe_schools")


def load_params():
    """Load parameters from config file."""
    params_path = CONFIGS_DIR / "params.yml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def download_doe_schools(params: dict) -> pd.DataFrame:
    """
    Download Bronx school locations from NYC Open Data.
    
    Uses the Socrata API (SODA) to query the school locations dataset.
    """
    dataset_id = params["nyc_open_data"]["school_locations_dataset"]
    base_url = params["nyc_open_data"]["base_url"]
    borough_code = params["geography"]["borough"]
    
    # Construct API URL
    # Filter for Bronx (borough = "X") and request JSON
    api_url = f"{base_url}/{dataset_id}.json"
    
    # Query parameters
    query_params = {
        "$where": f"borough = '{borough_code}'",
        "$limit": 10000,  # Get all schools (Bronx has ~400-500)
    }
    
    logger.info(f"Fetching schools from {api_url}")
    logger.info(f"Filter: borough = '{borough_code}'")
    
    response = requests.get(api_url, params=query_params, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    df = pd.DataFrame(data)
    
    logger.info(f"Downloaded {len(df)} school records")
    
    return df


def validate_data(df: pd.DataFrame, params: dict) -> bool:
    """Validate downloaded data meets expectations."""
    bounds = params["geography"]["bronx_bounds"]
    
    # Check we have data
    if len(df) == 0:
        logger.error("No schools downloaded!")
        return False
    
    # Check for required columns
    required_cols = ["system_code", "location_name", "latitude", "longitude"]
    # Column names may vary, check for alternatives
    available_cols = df.columns.tolist()
    logger.info(f"Available columns: {available_cols}")
    
    # Check for coordinate columns (may be named differently)
    lat_col = None
    lon_col = None
    for col in available_cols:
        if "lat" in col.lower():
            lat_col = col
        if "lon" in col.lower():
            lon_col = col
    
    if lat_col and lon_col:
        # Convert to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        
        # Check coordinates are within Bronx bounds
        valid_coords = (
            (df[lat_col] >= bounds["min_lat"]) & 
            (df[lat_col] <= bounds["max_lat"]) &
            (df[lon_col] >= bounds["min_lon"]) & 
            (df[lon_col] <= bounds["max_lon"])
        )
        
        invalid_count = (~valid_coords & df[lat_col].notna()).sum()
        if invalid_count > 0:
            logger.warning(f"{invalid_count} schools have coordinates outside Bronx bounds")
    
    # Check row count is reasonable (Bronx has ~400-500 public schools)
    if len(df) < 100:
        logger.warning(f"Only {len(df)} schools downloaded - seems low for Bronx")
    elif len(df) > 1000:
        logger.warning(f"{len(df)} schools downloaded - seems high, may include non-public schools")
    else:
        logger.info(f"School count ({len(df)}) is within expected range")
    
    return True


def main():
    """Main entry point."""
    log_step_start(logger, "download_doe_schools")
    
    try:
        # Load parameters
        params = load_params()
        
        # Download data
        df = download_doe_schools(params)
        
        # Validate
        if not validate_data(df, params):
            logger.error("Data validation failed")
            sys.exit(1)
        
        # Write output
        today = datetime.now().strftime("%Y-%m-%d")
        output_path = RAW_DIR / f"doe_school_locations_{today}.csv"
        
        atomic_write_csv(output_path, df)
        log_output_written(logger, output_path, row_count=len(df))
        
        # Write metadata sidecar
        write_metadata_sidecar(
            data_path=output_path,
            script_name="01_collect_doe_schools.py",
            run_id=get_run_id(),
            description="Bronx public school locations from NYC Open Data",
            inputs=[f"NYC Open Data API: dataset {params['nyc_open_data']['school_locations_dataset']}"],
            row_count=len(df),
            columns=list(df.columns),
        )
        
        # Update manifest
        source_url = f"{params['nyc_open_data']['base_url']}/{params['nyc_open_data']['school_locations_dataset']}"
        update_manifest(RAW_DIR, output_path.name, source_url, len(df))
        
        log_step_end(logger, "download_doe_schools", schools_downloaded=len(df))
        
        # Print summary
        print(f"\n✅ Successfully downloaded {len(df)} Bronx schools")
        print(f"   Output: {output_path}")
        
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

