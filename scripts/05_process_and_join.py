#!/usr/bin/env python3
"""
Script 05: Process and Join All Data Sources

Input:
    - data/raw/doe_school_locations_2025-12-25.csv
    - data/manual/mshp_school_list_expanded_2025-12-25.csv
    - data/raw/doe_attendance_2025-12-25.csv
    - data/raw/asthma_ed_visits_uhf_2025-12-25.csv
    - data/raw/asthma_hospitalizations_uhf_2025-12-25.csv
    - data/geo/uhf_42_neighborhoods.geojson

Output:
    - data/processed/bronx_schools_clean.csv
    - data/processed/bronx_schools_with_mshp_status.csv
    - data/processed/bronx_schools_with_attendance.csv
    - data/processed/bronx_schools_with_uhf.csv
    - data/processed/bronx_schools_analysis_ready.csv

This script performs all Phase 2 data processing:
1. Clean and standardize DOE school locations
2. Match MSHP schools to DOE records using DBNs
3. Join attendance data
4. Spatial join to assign UHF neighborhoods
5. Join asthma data
"""
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import RAW_DIR, PROCESSED_DIR, MANUAL_DIR, GEO_DIR
from mshp_gap.io_utils import atomic_write_csv, write_metadata_sidecar
from mshp_gap.logging_utils import (
    get_logger,
    get_run_id,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("05_process_and_join")


def load_doe_schools():
    """Load and clean DOE school locations."""
    log_step_start(logger, "load_doe_schools")
    
    # Find the most recent DOE schools file
    doe_files = list(RAW_DIR.glob("doe_school_locations_*.csv"))
    if not doe_files:
        raise FileNotFoundError("No DOE school locations file found in data/raw/")
    
    doe_file = sorted(doe_files)[-1]  # Most recent
    logger.info(f"Loading DOE schools from: {doe_file.name}")
    
    df = pd.read_csv(doe_file)
    logger.info(f"Loaded {len(df)} total school records")
    
    # Filter to Open schools only
    if 'status_descriptions' in df.columns:
        df = df[df['status_descriptions'] == 'Open'].copy()
        logger.info(f"Filtered to {len(df)} open schools")
    
    # Create standardized DBN column from system_code
    df['dbn'] = df['system_code'].str.strip()
    
    # Create standardized columns
    df['school_name'] = df['location_name'].str.strip()
    df['address'] = df['primary_address_line_1'].str.strip()
    df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['district'] = df['geographical_district_code'].astype(str)
    
    # Filter to valid coordinates
    df = df.dropna(subset=['lat', 'lon'])
    logger.info(f"After removing null coordinates: {len(df)} schools")
    
    # Validate coordinates are in Bronx area
    bronx_mask = (
        (df['lat'] >= 40.78) & (df['lat'] <= 40.92) &
        (df['lon'] >= -73.93) & (df['lon'] <= -73.75)
    )
    df = df[bronx_mask].copy()
    logger.info(f"After Bronx coordinate filter: {len(df)} schools")
    
    # Remove duplicate DBNs (source data has duplicates)
    dupe_count = len(df) - df['dbn'].nunique()
    if dupe_count > 0:
        logger.warning(f"Removing {dupe_count} duplicate DBNs from source data")
        df = df.drop_duplicates(subset=['dbn'], keep='first')
        logger.info(f"After deduplication: {len(df)} unique schools")
    
    # Determine school type from name
    def get_school_type(name):
        name_upper = name.upper()
        if 'HIGH SCHOOL' in name_upper or name_upper.startswith('H.S.'):
            return 'High School'
        elif 'MIDDLE' in name_upper or 'M.S.' in name_upper or 'I.S.' in name_upper or 'J.H.S.' in name_upper:
            return 'Middle School'
        elif 'P.S.' in name_upper or 'ELEMENTARY' in name_upper:
            if 'M.S.' in name_upper or '/M.S.' in name_upper:
                return 'K-8'
            return 'Elementary'
        elif 'CHARTER' in name_upper:
            return 'Charter'
        elif 'ACADEMY' in name_upper or 'PREP' in name_upper:
            return 'High School'
        else:
            return 'Other'
    
    df['school_type'] = df['school_name'].apply(get_school_type)
    
    # Select and rename columns for output
    output_cols = ['dbn', 'school_name', 'address', 'lat', 'lon', 'district', 
                   'school_type', 'grades_final_text']
    df = df[output_cols].copy()
    df.columns = ['dbn', 'school_name', 'address', 'lat', 'lon', 'district',
                  'school_type', 'grades']
    
    log_step_end(logger, "load_doe_schools", school_count=len(df))
    return df


def load_mshp_schools():
    """Load MSHP school list with DBNs."""
    log_step_start(logger, "load_mshp_schools")
    
    # Find the expanded MSHP file
    mshp_files = list(MANUAL_DIR.glob("mshp_school_list_expanded_*.csv"))
    if not mshp_files:
        # Fall back to original file
        mshp_files = list(MANUAL_DIR.glob("mshp_school_list_*.csv"))
    
    if not mshp_files:
        raise FileNotFoundError("No MSHP school list found in data/manual/")
    
    mshp_file = sorted(mshp_files)[-1]
    logger.info(f"Loading MSHP schools from: {mshp_file.name}")
    
    df = pd.read_csv(mshp_file)
    logger.info(f"Loaded {len(df)} MSHP schools")
    
    # Standardize DBN column
    if 'school_dbn' in df.columns:
        df['dbn'] = df['school_dbn'].str.strip()
    elif 'dbn' in df.columns:
        df['dbn'] = df['dbn'].str.strip()
    else:
        raise ValueError("MSHP file must have 'school_dbn' or 'dbn' column")
    
    log_step_end(logger, "load_mshp_schools", school_count=len(df))
    return df


def load_attendance_data():
    """Load chronic absenteeism data."""
    log_step_start(logger, "load_attendance_data")
    
    attendance_files = list(RAW_DIR.glob("doe_attendance_*.csv"))
    if not attendance_files:
        logger.warning("No attendance data file found")
        return None
    
    attendance_file = sorted(attendance_files)[-1]
    logger.info(f"Loading attendance from: {attendance_file.name}")
    
    df = pd.read_csv(attendance_file)
    logger.info(f"Loaded {len(df)} attendance records")
    
    # Standardize columns
    df['dbn'] = df['dbn'].str.strip()
    df['chronic_absenteeism_rate'] = pd.to_numeric(df['chronic_absenteeism_rate'], errors='coerce')
    df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')
    
    log_step_end(logger, "load_attendance_data", record_count=len(df))
    return df[['dbn', 'chronic_absenteeism_rate', 'enrollment']]


def load_uhf_boundaries():
    """Load UHF 42 neighborhood boundaries."""
    log_step_start(logger, "load_uhf_boundaries")
    
    uhf_files = list(GEO_DIR.glob("uhf_42_neighborhoods.geojson"))
    if not uhf_files:
        raise FileNotFoundError("UHF boundaries file not found in data/geo/")
    
    gdf = gpd.read_file(uhf_files[0])
    logger.info(f"Loaded {len(gdf)} UHF neighborhoods")
    
    # Standardize column names
    if 'GEOCODE' in gdf.columns:
        gdf['uhf_code'] = gdf['GEOCODE']
    if 'GEONAME' in gdf.columns:
        gdf['uhf_name'] = gdf['GEONAME']
    if 'BOROUGH' in gdf.columns:
        gdf['borough'] = gdf['BOROUGH']
    
    # Filter to Bronx (UHF codes 101-107)
    bronx_gdf = gdf[(gdf['uhf_code'] >= 101) & (gdf['uhf_code'] <= 107)].copy()
    logger.info(f"Filtered to {len(bronx_gdf)} Bronx UHF neighborhoods")
    
    # Ensure correct CRS
    if bronx_gdf.crs is None:
        bronx_gdf = bronx_gdf.set_crs("EPSG:4326")
    elif bronx_gdf.crs != "EPSG:4326":
        bronx_gdf = bronx_gdf.to_crs("EPSG:4326")
    
    log_step_end(logger, "load_uhf_boundaries", neighborhood_count=len(bronx_gdf))
    return bronx_gdf


def load_asthma_data():
    """Load asthma ED visits and hospitalizations by UHF."""
    log_step_start(logger, "load_asthma_data")
    
    # Load ED visits
    ed_files = list(RAW_DIR.glob("asthma_ed_visits_uhf_*.csv"))
    hosp_files = list(RAW_DIR.glob("asthma_hospitalizations_uhf_*.csv"))
    
    asthma_df = None
    
    if ed_files:
        ed_df = pd.read_csv(sorted(ed_files)[-1])
        ed_df = ed_df[['uhf_code', 'rate_per_10000']].copy()
        ed_df.columns = ['uhf_code', 'asthma_ed_rate']
        asthma_df = ed_df
        logger.info(f"Loaded asthma ED visit rates for {len(ed_df)} UHF neighborhoods")
    
    if hosp_files:
        hosp_df = pd.read_csv(sorted(hosp_files)[-1])
        hosp_df = hosp_df[['uhf_code', 'rate_per_10000']].copy()
        hosp_df.columns = ['uhf_code', 'asthma_hosp_rate']
        
        if asthma_df is not None:
            asthma_df = asthma_df.merge(hosp_df, on='uhf_code', how='outer')
        else:
            asthma_df = hosp_df
        logger.info(f"Loaded asthma hospitalization rates")
    
    if asthma_df is None:
        logger.warning("No asthma data files found")
        return None
    
    # Filter to Bronx UHF codes (101-107)
    asthma_df = asthma_df[(asthma_df['uhf_code'] >= 101) & (asthma_df['uhf_code'] <= 107)].copy()
    logger.info(f"Bronx UHF asthma data: {len(asthma_df)} neighborhoods")
    
    log_step_end(logger, "load_asthma_data", neighborhood_count=len(asthma_df))
    return asthma_df


def match_mshp_to_doe(doe_df, mshp_df):
    """Match MSHP schools to DOE records using DBN."""
    log_step_start(logger, "match_mshp_to_doe")
    
    # Get set of MSHP DBNs
    mshp_dbns = set(mshp_df['dbn'].unique())
    logger.info(f"MSHP schools to match: {len(mshp_dbns)}")
    
    # Match by DBN
    doe_df['has_mshp'] = doe_df['dbn'].isin(mshp_dbns)
    
    matched_count = doe_df['has_mshp'].sum()
    logger.info(f"Matched {matched_count} schools with MSHP coverage")
    
    # Find unmatched MSHP schools (in MSHP list but not in DOE)
    doe_dbns = set(doe_df['dbn'].unique())
    unmatched_mshp = mshp_dbns - doe_dbns
    if unmatched_mshp:
        logger.warning(f"Unmatched MSHP DBNs (not in DOE list): {len(unmatched_mshp)}")
        for dbn in sorted(unmatched_mshp)[:10]:
            logger.warning(f"  - {dbn}")
    
    log_step_end(logger, "match_mshp_to_doe", matched=matched_count, unmatched=len(unmatched_mshp))
    return doe_df


def join_attendance(schools_df, attendance_df):
    """Join attendance data to schools."""
    log_step_start(logger, "join_attendance")
    
    if attendance_df is None:
        logger.warning("No attendance data to join")
        schools_df['chronic_absenteeism_rate'] = None
        schools_df['enrollment'] = None
        return schools_df
    
    before_count = len(schools_df)
    
    # Left join on DBN
    merged = schools_df.merge(
        attendance_df,
        on='dbn',
        how='left'
    )
    
    matched = merged['chronic_absenteeism_rate'].notna().sum()
    logger.info(f"Joined attendance data: {matched}/{before_count} schools have data")
    
    log_step_end(logger, "join_attendance", matched=matched, total=before_count)
    return merged


def spatial_join_uhf(schools_df, uhf_gdf):
    """Assign each school to its UHF neighborhood using spatial join."""
    log_step_start(logger, "spatial_join_uhf")
    
    # Create GeoDataFrame from schools
    geometry = [Point(lon, lat) for lon, lat in zip(schools_df['lon'], schools_df['lat'])]
    schools_gdf = gpd.GeoDataFrame(schools_df, geometry=geometry, crs="EPSG:4326")
    
    # Perform spatial join
    joined = gpd.sjoin(
        schools_gdf,
        uhf_gdf[['uhf_code', 'uhf_name', 'geometry']],
        how='left',
        predicate='within'
    )
    
    matched = joined['uhf_code'].notna().sum()
    logger.info(f"Spatial join: {matched}/{len(schools_df)} schools assigned to UHF")
    
    # Handle schools outside UHF boundaries (shouldn't happen if data is correct)
    unmatched = joined['uhf_code'].isna().sum()
    if unmatched > 0:
        logger.warning(f"{unmatched} schools not matched to any UHF neighborhood")
    
    # Drop geometry and index_right columns
    result = pd.DataFrame(joined.drop(columns=['geometry', 'index_right'], errors='ignore'))
    
    log_step_end(logger, "spatial_join_uhf", matched=matched, unmatched=unmatched)
    return result


def join_asthma_data(schools_df, asthma_df):
    """Join asthma rates to schools via UHF code."""
    log_step_start(logger, "join_asthma_data")
    
    if asthma_df is None:
        logger.warning("No asthma data to join")
        schools_df['asthma_ed_rate'] = None
        schools_df['asthma_hosp_rate'] = None
        return schools_df
    
    before_count = len(schools_df)
    
    # Left join on UHF code
    merged = schools_df.merge(
        asthma_df,
        on='uhf_code',
        how='left'
    )
    
    matched = merged['asthma_ed_rate'].notna().sum()
    logger.info(f"Joined asthma data: {matched}/{before_count} schools have asthma rates")
    
    log_step_end(logger, "join_asthma_data", matched=matched, total=before_count)
    return merged


def main():
    """Main entry point for Phase 2 data processing."""
    logger.info("=" * 60)
    logger.info("PHASE 2: DATA PROCESSING")
    logger.info("=" * 60)
    
    run_id = get_run_id()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Step 1: Load and clean DOE schools
    logger.info("\n--- Step 1: Load and clean DOE school locations ---")
    doe_df = load_doe_schools()
    
    # Save cleaned schools
    output_path = PROCESSED_DIR / "bronx_schools_clean.csv"
    atomic_write_csv(output_path, doe_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="05_process_and_join.py",
        run_id=run_id,
        description="Cleaned Bronx public school locations",
        inputs=["data/raw/doe_school_locations_*.csv"],
        row_count=len(doe_df),
        columns=list(doe_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(doe_df))
    
    # Step 2: Load MSHP schools and match
    logger.info("\n--- Step 2: Match MSHP schools to DOE records ---")
    mshp_df = load_mshp_schools()
    doe_df = match_mshp_to_doe(doe_df, mshp_df)
    
    # Save with MSHP status
    output_path = PROCESSED_DIR / "bronx_schools_with_mshp_status.csv"
    atomic_write_csv(output_path, doe_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="05_process_and_join.py",
        run_id=run_id,
        description="Bronx schools with MSHP coverage status",
        inputs=["bronx_schools_clean.csv", "mshp_school_list_expanded_*.csv"],
        row_count=len(doe_df),
        columns=list(doe_df.columns),
        mshp_covered=int(doe_df['has_mshp'].sum()),
        mshp_not_covered=int((~doe_df['has_mshp']).sum()),
    )
    log_output_written(logger, output_path, row_count=len(doe_df))
    
    # Step 3: Join attendance data
    logger.info("\n--- Step 3: Join attendance data ---")
    attendance_df = load_attendance_data()
    doe_df = join_attendance(doe_df, attendance_df)
    
    # Save with attendance
    output_path = PROCESSED_DIR / "bronx_schools_with_attendance.csv"
    atomic_write_csv(output_path, doe_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="05_process_and_join.py",
        run_id=run_id,
        description="Bronx schools with MSHP status and attendance data",
        inputs=["bronx_schools_with_mshp_status.csv", "doe_attendance_*.csv"],
        row_count=len(doe_df),
        columns=list(doe_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(doe_df))
    
    # Step 4: Spatial join to UHF neighborhoods
    logger.info("\n--- Step 4: Assign schools to UHF neighborhoods ---")
    uhf_gdf = load_uhf_boundaries()
    doe_df = spatial_join_uhf(doe_df, uhf_gdf)
    
    # Save with UHF
    output_path = PROCESSED_DIR / "bronx_schools_with_uhf.csv"
    atomic_write_csv(output_path, doe_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="05_process_and_join.py",
        run_id=run_id,
        description="Bronx schools with UHF neighborhood assignments",
        inputs=["bronx_schools_with_attendance.csv", "uhf_42_neighborhoods.geojson"],
        row_count=len(doe_df),
        columns=list(doe_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(doe_df))
    
    # Step 5: Join asthma data
    logger.info("\n--- Step 5: Join asthma data ---")
    asthma_df = load_asthma_data()
    doe_df = join_asthma_data(doe_df, asthma_df)
    
    # Save final analysis-ready file
    output_path = PROCESSED_DIR / "bronx_schools_analysis_ready.csv"
    atomic_write_csv(output_path, doe_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="05_process_and_join.py",
        run_id=run_id,
        description="Analysis-ready Bronx schools with all data joined",
        inputs=[
            "doe_school_locations_*.csv",
            "mshp_school_list_expanded_*.csv",
            "doe_attendance_*.csv",
            "uhf_42_neighborhoods.geojson",
            "asthma_ed_visits_uhf_*.csv",
            "asthma_hospitalizations_uhf_*.csv",
        ],
        row_count=len(doe_df),
        columns=list(doe_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(doe_df))
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COMPLETE - SUMMARY")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("PHASE 2: DATA PROCESSING - COMPLETE")
    print("=" * 60)
    print(f"\nTotal Bronx schools: {len(doe_df)}")
    print(f"MSHP covered: {doe_df['has_mshp'].sum()}")
    print(f"NOT covered: {(~doe_df['has_mshp']).sum()}")
    print(f"\nWith attendance data: {doe_df['chronic_absenteeism_rate'].notna().sum()}")
    print(f"With UHF assignment: {doe_df['uhf_code'].notna().sum()}")
    print(f"With asthma data: {doe_df['asthma_ed_rate'].notna().sum()}")
    
    print("\n--- MSHP Coverage by School Type ---")
    coverage_by_type = doe_df.groupby('school_type')['has_mshp'].agg(['sum', 'count'])
    coverage_by_type['pct'] = (coverage_by_type['sum'] / coverage_by_type['count'] * 100).round(1)
    coverage_by_type.columns = ['MSHP Covered', 'Total', 'Coverage %']
    print(coverage_by_type.to_string())
    
    print("\n--- Schools by UHF Neighborhood ---")
    if 'uhf_name' in doe_df.columns:
        uhf_summary = doe_df.groupby('uhf_name').agg({
            'dbn': 'count',
            'has_mshp': 'sum',
            'asthma_ed_rate': 'first'
        }).round(1)
        uhf_summary.columns = ['Total Schools', 'MSHP Covered', 'Asthma ED Rate']
        uhf_summary['Coverage %'] = (uhf_summary['MSHP Covered'] / uhf_summary['Total Schools'] * 100).round(1)
        uhf_summary = uhf_summary.sort_values('Asthma ED Rate', ascending=False)
        print(uhf_summary.to_string())
    
    print(f"\n✅ Output saved to: {PROCESSED_DIR / 'bronx_schools_analysis_ready.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

