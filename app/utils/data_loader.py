"""
Data loading and caching utilities for the MSHP Coverage Gap Dashboard.
"""
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import yaml


# Resolve paths relative to app directory
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"

def _add_distance_to_nearest_mshp_miles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `distance_to_nearest_mshp_miles` if missing.

    Uses a small-area approximation (Bronx) by scaling lat/lon degrees to miles.
    """
    if 'distance_to_nearest_mshp_miles' in df.columns:
        return df

    required_cols = {'has_mshp', 'latitude', 'longitude'}
    if not required_cols.issubset(df.columns):
        return df

    mshp = df[df['has_mshp']].copy()
    if mshp.empty:
        df['distance_to_nearest_mshp_miles'] = np.nan
        return df

    # Approx miles per degree near NYC latitude
    lat_miles = 69.0
    lon_miles = 53.0

    all_coords = df[['latitude', 'longitude']].to_numpy(dtype=float, copy=True)
    mshp_coords = mshp[['latitude', 'longitude']].to_numpy(dtype=float, copy=True)

    # Scale to approximate miles (Euclidean ok at this scale)
    all_coords[:, 0] *= lat_miles
    all_coords[:, 1] *= lon_miles
    mshp_coords[:, 0] *= lat_miles
    mshp_coords[:, 1] *= lon_miles

    # Broadcast distances: (n, 1, 2) - (1, m, 2) -> (n, m, 2)
    diffs = all_coords[:, None, :] - mshp_coords[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    min_dists = dists.min(axis=1)

    df['distance_to_nearest_mshp_miles'] = min_dists
    df.loc[df['has_mshp'], 'distance_to_nearest_mshp_miles'] = 0.0
    return df


@st.cache_data(ttl=3600)
def load_schools_data() -> pd.DataFrame:
    """
    Load the main schools dataset with all analysis columns.
    Cached for 1 hour.
    """
    candidates = [
        PROCESSED_DIR / "bronx_schools_full.csv",
        PROCESSED_DIR / "bronx_schools_enhanced.csv",
        PROCESSED_DIR / "bronx_schools_analysis_ready.csv",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    if data_path is None:
        st.error(
            "Processed schools dataset not found.\n\n"
            "Expected one of:\n"
            "- `data/processed/bronx_schools_full.csv`\n"
            "- `data/processed/bronx_schools_enhanced.csv`\n"
            "- `data/processed/bronx_schools_analysis_ready.csv`\n\n"
            "Run the pipeline to generate it (see `README.md`)."
        )
        st.stop()

    df = pd.read_csv(data_path)
    
    # Ensure boolean type for has_mshp (robust to 0/1, True/False, and string values)
    if df['has_mshp'].dtype == object:
        df['has_mshp'] = (
            df['has_mshp']
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['true', '1', 'yes', 'y'])
        )
    else:
        df['has_mshp'] = df['has_mshp'].astype(bool)
    
    # Create display-friendly columns
    df['mshp_status'] = df['has_mshp'].map({True: 'MSHP Covered', False: 'Not Covered'})
    
    # Fill missing values for display
    df['enrollment'] = df['enrollment'].fillna(0).astype(int)
    df['chronic_absenteeism_rate'] = df['chronic_absenteeism_rate'].fillna(df['chronic_absenteeism_rate'].median())
    
    # Standardize coordinate column names
    if 'lat' in df.columns and 'latitude' not in df.columns:
        df['latitude'] = df['lat']
    if 'lon' in df.columns and 'longitude' not in df.columns:
        df['longitude'] = df['lon']

    # Add accessibility metric if missing
    df = _add_distance_to_nearest_mshp_miles(df)
    
    # Add priority tier and score from priority rankings if available
    if 'priority_tier' not in df.columns:
        try:
            priority_df = pd.read_csv(TABLES_DIR / "non_mshp_schools_priority_ranked.csv")
            if 'priority_score' in priority_df.columns and 'priority_tier' in priority_df.columns:
                df = df.merge(
                    priority_df[['dbn', 'priority_score', 'priority_tier']],
                    on='dbn',
                    how='left'
                )
        except FileNotFoundError:
            # Calculate priority tier from available data
            non_mshp = df[~df['has_mshp']].copy()
            if len(non_mshp) > 0:
                # Simple priority score based on available metrics
                from scipy.stats import percentileofscore
                
                # Normalize key metrics
                df['priority_score'] = 0.0
                non_mshp_idx = ~df['has_mshp']
                
                if 'asthma_ed_rate' in df.columns:
                    asthma_pct = df.loc[non_mshp_idx, 'asthma_ed_rate'].rank(pct=True) * 100
                    df.loc[non_mshp_idx, 'priority_score'] += asthma_pct * 0.4
                
                if 'chronic_absenteeism_rate' in df.columns:
                    absent_pct = df.loc[non_mshp_idx, 'chronic_absenteeism_rate'].rank(pct=True) * 100
                    df.loc[non_mshp_idx, 'priority_score'] += absent_pct * 0.4
                
                if 'enrollment' in df.columns:
                    enroll_pct = df.loc[non_mshp_idx, 'enrollment'].rank(pct=True) * 100
                    df.loc[non_mshp_idx, 'priority_score'] += enroll_pct * 0.2
                
                # Assign tiers
                df['priority_tier'] = None
                score_80 = df.loc[non_mshp_idx, 'priority_score'].quantile(0.80)
                score_60 = df.loc[non_mshp_idx, 'priority_score'].quantile(0.60)
                score_40 = df.loc[non_mshp_idx, 'priority_score'].quantile(0.40)
                
                df.loc[non_mshp_idx & (df['priority_score'] >= score_80), 'priority_tier'] = 'Tier 1 - Critical'
                df.loc[non_mshp_idx & (df['priority_score'] >= score_60) & (df['priority_score'] < score_80), 'priority_tier'] = 'Tier 2 - High'
                df.loc[non_mshp_idx & (df['priority_score'] >= score_40) & (df['priority_score'] < score_60), 'priority_tier'] = 'Tier 3 - Moderate'
                df.loc[non_mshp_idx & (df['priority_score'] < score_40), 'priority_tier'] = 'Tier 4 - Lower'
    
    return df


@st.cache_data(ttl=3600)
def load_uhf_boundaries() -> gpd.GeoDataFrame:
    """Load UHF neighborhood boundaries."""
    gdf = gpd.read_file(GEO_DIR / "uhf_42_neighborhoods.geojson")
    
    # Filter to Bronx neighborhoods
    bronx_uhf_names = [
        'Kingsbridge - Riverdale', 'Northeast Bronx', 'Fordham - Bronx Park',
        'Pelham - Throgs Neck', 'Crotona - Tremont', 'High Bridge - Morrisania',
        'Hunts Point - Mott Haven'
    ]
    
    # Try different column names for UHF name
    name_col = None
    for col in ['uhf_neigh', 'GEONAME', 'uhf_name', 'name']:
        if col in gdf.columns:
            name_col = col
            break
    
    if name_col:
        gdf = gdf[gdf[name_col].isin(bronx_uhf_names)].copy()
        gdf = gdf.rename(columns={name_col: 'uhf_name'})
    
    return gdf


@st.cache_data(ttl=3600)
def load_params() -> dict:
    """Load configuration parameters."""
    with open(CONFIGS_DIR / "params.yml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=3600)
def load_priority_rankings() -> pd.DataFrame:
    """Load priority-ranked non-MSHP schools."""
    try:
        return pd.read_csv(TABLES_DIR / "non_mshp_schools_priority_ranked.csv")
    except FileNotFoundError:
        # Fall back to calculating from main data
        df = load_schools_data()
        return df[~df['has_mshp']].sort_values('priority_score', ascending=False)


@st.cache_data(ttl=3600)
def load_equity_data() -> dict:
    """Load all equity analysis outputs."""
    equity_data = {}
    
    tables = [
        'equity_demographic_disparities',
        'equity_intersectional_risk',
        'equity_by_school_type',
        'equity_geographic',
        'high_risk_uncovered_schools',
        'health_coverage_mismatch',
        'neighborhood_health_rankings',
    ]
    
    for table in tables:
        try:
            equity_data[table] = pd.read_csv(TABLES_DIR / f"{table}.csv")
        except FileNotFoundError:
            equity_data[table] = None
    
    return equity_data


@st.cache_data(ttl=3600)
def load_expansion_scenarios() -> dict:
    """Load optimal expansion scenario data."""
    scenarios = {}
    
    for n in [5, 10, 20]:
        try:
            df = pd.read_csv(TABLES_DIR / f"optimal_expansion_{n}_schools.csv")
            # Backwards/forwards compatibility: normalize method column name
            if 'method' not in df.columns and 'selection_method' in df.columns:
                df['method'] = df['selection_method']
            scenarios[n] = df
        except FileNotFoundError:
            scenarios[n] = None
    
    try:
        scenarios['comparison'] = pd.read_csv(TABLES_DIR / "expansion_scenarios_comparison.csv")
    except FileNotFoundError:
        scenarios['comparison'] = None
    
    return scenarios


def get_neighborhood_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate neighborhood-level summary statistics."""
    summary = df.groupby('uhf_name').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'enrollment': 'sum',
        'chronic_absenteeism_rate': 'mean',
        'asthma_ed_rate': 'first',
        'pct_poverty': 'mean',
    }).reset_index()
    
    # Flatten column names
    summary.columns = [
        'uhf_name', 'total_schools', 'mshp_count', 'coverage_rate',
        'total_students', 'avg_absenteeism', 'asthma_rate', 'avg_poverty'
    ]
    
    summary['coverage_pct'] = summary['coverage_rate'] * 100
    summary['non_mshp_count'] = summary['total_schools'] - summary['mshp_count']
    
    return summary.sort_values('asthma_rate', ascending=False)


@st.cache_data(ttl=3600)
def load_regression_results() -> pd.DataFrame | None:
    """Load regression results table (if available)."""
    try:
        return pd.read_csv(TABLES_DIR / "regression_results.csv")
    except FileNotFoundError:
        return None


@st.cache_data(ttl=3600)
def load_regression_summary_text() -> str | None:
    """Load the full regression summary text (if available)."""
    try:
        return (TABLES_DIR / "regression_full_summary.txt").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def get_school_types(df: pd.DataFrame) -> list:
    """Get unique school types for filtering."""
    if 'school_type' in df.columns:
        return sorted(df['school_type'].dropna().unique().tolist())
    return []


def get_districts(df: pd.DataFrame) -> list:
    """Get unique districts for filtering."""
    if 'district' in df.columns:
        return sorted(df['district'].dropna().unique().tolist())
    return []

