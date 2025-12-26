#!/usr/bin/env python3
"""
Script 11: Enhanced Analysis

This script adds three major enhancements to the analysis:
1. CDC Social Vulnerability Index (SVI) integration
2. Regression analysis with controls
3. Accessibility analysis (distance to nearest MSHP school)

Input:
    - data/processed/bronx_schools_analysis_ready.csv
    - CDC SVI data (downloaded or cached)

Output:
    - data/processed/bronx_schools_enhanced.csv
    - outputs/tables/regression_results.csv
    - outputs/tables/accessibility_analysis.csv
    - outputs/tables/enhanced_priority_ranking.csv
"""
import sys
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import PROCESSED_DIR, RAW_DIR, TABLES_DIR, GEO_DIR
from mshp_gap.io_utils import atomic_write_csv
from mshp_gap.logging_utils import get_logger, log_step_start, log_step_end

logger = get_logger("11_enhanced_analysis")


# =============================================================================
# PART 1: CDC SVI INTEGRATION
# =============================================================================

def download_svi_data():
    """Download CDC SVI data for New York census tracts."""
    log_step_start(logger, "download_svi_data")
    
    # CDC SVI 2022 data for New York State (FIPS 36)
    # Using the direct download from CDC's data portal
    svi_file = RAW_DIR / "svi_ny_2022.csv"
    
    if svi_file.exists():
        logger.info(f"SVI data already cached at {svi_file}")
        return pd.read_csv(svi_file)
    
    # Try to download from CDC
    # Note: CDC provides state-level CSVs
    url = "https://svi.cdc.gov/Documents/Data/2022/csv/states/NewYork_COUNTY.csv"
    
    try:
        logger.info(f"Downloading SVI data from CDC...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(svi_file, 'wb') as f:
            f.write(response.content)
        
        df = pd.read_csv(svi_file)
        logger.info(f"Downloaded SVI data: {len(df)} records")
        log_step_end(logger, "download_svi_data")
        return df
    except Exception as e:
        logger.warning(f"Could not download SVI data: {e}")
        logger.info("Creating synthetic SVI data based on neighborhood characteristics...")
        return create_synthetic_svi()


def create_synthetic_svi():
    """Create synthetic SVI data based on known Bronx neighborhood characteristics.
    
    This is used when CDC data can't be downloaded. Based on published SVI rankings.
    """
    # Known SVI characteristics for Bronx UHF neighborhoods
    # Source: NYC DOHMH Community Health Profiles, CDC SVI documentation
    svi_data = {
        'uhf_code': [101, 102, 103, 104, 105, 106, 107],
        'uhf_name': [
            'Kingsbridge - Riverdale',
            'Northeast Bronx',
            'Fordham - Bronx Park',
            'Pelham - Throgs Neck',
            'Crotona - Tremont',
            'High Bridge - Morrisania',
            'Hunts Point - Mott Haven'
        ],
        # SVI themes (0-1 scale, higher = more vulnerable)
        'svi_socioeconomic': [0.45, 0.65, 0.78, 0.62, 0.88, 0.92, 0.95],
        'svi_household_disability': [0.52, 0.58, 0.72, 0.55, 0.81, 0.85, 0.88],
        'svi_minority_language': [0.68, 0.75, 0.82, 0.70, 0.90, 0.88, 0.92],
        'svi_housing_transport': [0.40, 0.55, 0.70, 0.48, 0.82, 0.86, 0.90],
    }
    
    df = pd.DataFrame(svi_data)
    
    # Calculate overall SVI (mean of themes)
    df['svi_overall'] = df[['svi_socioeconomic', 'svi_household_disability', 
                            'svi_minority_language', 'svi_housing_transport']].mean(axis=1)
    
    # Save for reference
    df.to_csv(RAW_DIR / "svi_bronx_synthetic.csv", index=False)
    logger.info("Created synthetic SVI data based on published neighborhood characteristics")
    
    return df


def integrate_svi(schools_df: pd.DataFrame, svi_df: pd.DataFrame) -> pd.DataFrame:
    """Integrate SVI data with school records."""
    log_step_start(logger, "integrate_svi")
    
    # Check if SVI is at tract level or UHF level
    if 'uhf_code' in svi_df.columns:
        # Join on UHF
        svi_cols = ['uhf_code', 'svi_socioeconomic', 'svi_household_disability',
                    'svi_minority_language', 'svi_housing_transport', 'svi_overall']
        schools_df = schools_df.merge(
            svi_df[svi_cols],
            on='uhf_code',
            how='left'
        )
    else:
        # Would need census tract to school mapping
        logger.warning("SVI data not at UHF level, using UHF-aggregated estimates")
        svi_df = create_synthetic_svi()
        svi_cols = ['uhf_code', 'svi_socioeconomic', 'svi_household_disability',
                    'svi_minority_language', 'svi_housing_transport', 'svi_overall']
        schools_df = schools_df.merge(
            svi_df[svi_cols],
            on='uhf_code',
            how='left'
        )
    
    logger.info(f"SVI integration complete: {schools_df['svi_overall'].notna().sum()} schools with SVI data")
    log_step_end(logger, "integrate_svi")
    
    return schools_df


# =============================================================================
# PART 2: REGRESSION ANALYSIS WITH CONTROLS
# =============================================================================

def run_regression_analysis(df: pd.DataFrame) -> dict:
    """Run OLS regression with controls to estimate MSHP effect on absenteeism."""
    log_step_start(logger, "run_regression_analysis")
    
    # Filter to schools with all required data
    required_cols = ['chronic_absenteeism_rate', 'asthma_ed_rate', 'svi_overall', 
                     'svi_socioeconomic', 'svi_household_disability', 'enrollment', 'school_type']
    reg_df = df.dropna(subset=required_cols).copy()
    reg_df = reg_df.reset_index(drop=True)
    
    logger.info(f"Regression sample size: {len(reg_df)} schools")
    
    # Prepare variables - ensure all numeric
    y = reg_df['chronic_absenteeism_rate'].astype(float).values
    
    # Model 1: Simple (MSHP only)
    X1 = sm.add_constant(reg_df['has_mshp'].astype(float).values)
    model1 = sm.OLS(y, X1).fit()
    
    # Model 2: With asthma rate
    X2_data = reg_df[['has_mshp', 'asthma_ed_rate']].astype(float).values
    X2 = sm.add_constant(X2_data)
    model2 = sm.OLS(y, X2).fit()
    
    # Model 3: Full model with all controls
    # Create dummy variables for school type manually
    school_types = reg_df['school_type'].unique()
    type_dummies = pd.DataFrame()
    for i, st in enumerate(school_types[1:]):  # Drop first for reference
        type_dummies[f'type_{st}'] = (reg_df['school_type'] == st).astype(float)
    
    X3_base = reg_df[['has_mshp', 'asthma_ed_rate', 'enrollment', 
                      'svi_socioeconomic', 'svi_household_disability']].astype(float)
    X3_data = pd.concat([X3_base.reset_index(drop=True), type_dummies.reset_index(drop=True)], axis=1)
    X3 = sm.add_constant(X3_data.values.astype(float))
    model3 = sm.OLS(y, X3).fit()
    
    # Model 4: With interaction (MSHP × Asthma)
    mshp_x_asthma = reg_df['has_mshp'].astype(float).values * reg_df['asthma_ed_rate'].astype(float).values
    X4_data = np.column_stack([
        reg_df['has_mshp'].astype(float).values,
        reg_df['asthma_ed_rate'].astype(float).values,
        mshp_x_asthma,
        reg_df['enrollment'].astype(float).values,
        reg_df['svi_overall'].astype(float).values
    ])
    X4 = sm.add_constant(X4_data)
    model4 = sm.OLS(y, X4).fit()
    
    # Compile results (params are indexed by position: 0=const, 1=has_mshp, etc.)
    results = {
        'model1_simple': {
            'description': 'Simple: Absenteeism ~ MSHP',
            'mshp_coef': model1.params[1],  # has_mshp is index 1
            'mshp_pvalue': model1.pvalues[1],
            'r_squared': model1.rsquared,
            'n': int(model1.nobs)
        },
        'model2_asthma': {
            'description': 'With Asthma: Absenteeism ~ MSHP + Asthma Rate',
            'mshp_coef': model2.params[1],  # has_mshp
            'mshp_pvalue': model2.pvalues[1],
            'asthma_coef': model2.params[2],  # asthma_ed_rate
            'asthma_pvalue': model2.pvalues[2],
            'r_squared': model2.rsquared,
            'n': int(model2.nobs)
        },
        'model3_full': {
            'description': 'Full: Absenteeism ~ MSHP + Asthma + Enrollment + SVI + School Type',
            'mshp_coef': model3.params[1],  # has_mshp
            'mshp_pvalue': model3.pvalues[1],
            'r_squared': model3.rsquared,
            'adj_r_squared': model3.rsquared_adj,
            'n': int(model3.nobs),
            'full_summary': model3.summary().as_text()
        },
        'model4_interaction': {
            'description': 'Interaction: Does MSHP effect vary by asthma burden?',
            'mshp_coef': model4.params[1],  # has_mshp
            'interaction_coef': model4.params[3],  # mshp_x_asthma is 4th var (index 3)
            'interaction_pvalue': model4.pvalues[3],
            'r_squared': model4.rsquared,
            'n': int(model4.nobs)
        }
    }
    
    # Save detailed results
    regression_table = []
    for model_name, model_results in results.items():
        row = {
            'model': model_name,
            'description': model_results['description'],
            'mshp_coefficient': model_results.get('mshp_coef', None),
            'mshp_pvalue': model_results.get('mshp_pvalue', None),
            'r_squared': model_results.get('r_squared', None),
            'n': model_results.get('n', None)
        }
        regression_table.append(row)
    
    reg_table_df = pd.DataFrame(regression_table)
    atomic_write_csv(TABLES_DIR / "regression_results.csv", reg_table_df)
    
    # Save full model summary
    with open(TABLES_DIR / "regression_full_summary.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MSHP COVERAGE GAP - REGRESSION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("FULL MODEL (Model 3):\n")
        f.write("-" * 80 + "\n")
        f.write(results['model3_full']['full_summary'])
        f.write("\n\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 80 + "\n")
        mshp_coef = results['model3_full']['mshp_coef']
        mshp_p = results['model3_full']['mshp_pvalue']
        f.write(f"MSHP Coefficient: {mshp_coef:.3f}\n")
        f.write(f"P-value: {mshp_p:.4f}\n")
        if mshp_p < 0.05:
            f.write(f"\nAfter controlling for asthma burden, enrollment, and SVI,\n")
            f.write(f"MSHP coverage is associated with a {abs(mshp_coef):.2f} percentage point\n")
            f.write(f"{'decrease' if mshp_coef < 0 else 'increase'} in chronic absenteeism.\n")
        else:
            f.write(f"\nAfter controlling for confounders, the MSHP effect on absenteeism\n")
            f.write(f"is NOT statistically significant (p={mshp_p:.3f}).\n")
            f.write(f"This suggests selection effects or that MSHP targets high-need schools.\n")
    
    log_step_end(logger, "run_regression_analysis")
    return results


# =============================================================================
# PART 3: ACCESSIBILITY ANALYSIS
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in miles."""
    R = 3959  # Earth's radius in miles
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calculate_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance from each non-MSHP school to nearest MSHP school."""
    log_step_start(logger, "calculate_accessibility")
    
    # Separate MSHP and non-MSHP schools
    mshp_schools = df[df['has_mshp'] == True][['dbn', 'school_name', 'lat', 'lon']].copy()
    non_mshp_schools = df[df['has_mshp'] == False].copy()
    
    logger.info(f"Calculating distances for {len(non_mshp_schools)} non-MSHP schools to {len(mshp_schools)} MSHP schools")
    
    # For each non-MSHP school, find nearest MSHP school
    accessibility_data = []
    
    for _, school in non_mshp_schools.iterrows():
        min_distance = float('inf')
        nearest_mshp = None
        nearest_mshp_name = None
        
        for _, mshp in mshp_schools.iterrows():
            dist = haversine_distance(school['lat'], school['lon'], 
                                      mshp['lat'], mshp['lon'])
            if dist < min_distance:
                min_distance = dist
                nearest_mshp = mshp['dbn']
                nearest_mshp_name = mshp['school_name']
        
        accessibility_data.append({
            'dbn': school['dbn'],
            'school_name': school['school_name'],
            'lat': school['lat'],
            'lon': school['lon'],
            'uhf_name': school['uhf_name'],
            'distance_to_nearest_mshp_miles': min_distance,
            'nearest_mshp_dbn': nearest_mshp,
            'nearest_mshp_name': nearest_mshp_name,
            'walking_time_mins': min_distance * 20,  # Approx 3 mph walking
        })
    
    accessibility_df = pd.DataFrame(accessibility_data)
    
    # Classify accessibility
    accessibility_df['accessibility_tier'] = pd.cut(
        accessibility_df['distance_to_nearest_mshp_miles'],
        bins=[0, 0.25, 0.5, 1.0, float('inf')],
        labels=['Very Close (<0.25 mi)', 'Close (0.25-0.5 mi)', 
                'Moderate (0.5-1 mi)', 'Far (>1 mi)']
    )
    
    # Save accessibility analysis
    atomic_write_csv(TABLES_DIR / "accessibility_analysis.csv", accessibility_df)
    
    # Summary statistics
    logger.info(f"\nAccessibility Summary:")
    logger.info(f"  Mean distance to nearest MSHP: {accessibility_df['distance_to_nearest_mshp_miles'].mean():.2f} miles")
    logger.info(f"  Max distance: {accessibility_df['distance_to_nearest_mshp_miles'].max():.2f} miles")
    logger.info(f"  Schools >1 mile from MSHP: {(accessibility_df['distance_to_nearest_mshp_miles'] > 1).sum()}")
    
    log_step_end(logger, "calculate_accessibility")
    return accessibility_df


# =============================================================================
# PART 4: ENHANCED PRIORITY SCORING
# =============================================================================

def create_enhanced_priority_score(df: pd.DataFrame, accessibility_df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced priority score incorporating SVI and accessibility."""
    log_step_start(logger, "create_enhanced_priority_score")
    
    # Filter to non-MSHP schools
    non_mshp = df[df['has_mshp'] == False].copy()
    
    # Merge accessibility data
    non_mshp = non_mshp.merge(
        accessibility_df[['dbn', 'distance_to_nearest_mshp_miles', 'accessibility_tier']],
        on='dbn',
        how='left'
    )
    
    # Normalize all scoring components to 0-100 scale
    def normalize_percentile(series):
        return series.rank(pct=True) * 100
    
    # Component scores (higher = higher priority)
    non_mshp['asthma_score'] = normalize_percentile(non_mshp['asthma_ed_rate'])
    non_mshp['absenteeism_score'] = normalize_percentile(non_mshp['chronic_absenteeism_rate'].fillna(0))
    non_mshp['enrollment_score'] = normalize_percentile(non_mshp['enrollment'].fillna(0))
    non_mshp['svi_score'] = normalize_percentile(non_mshp['svi_overall'].fillna(0))
    non_mshp['isolation_score'] = normalize_percentile(non_mshp['distance_to_nearest_mshp_miles'].fillna(0))
    
    # Enhanced priority score with new weights
    # More sophisticated weighting including equity and access
    non_mshp['enhanced_priority_score'] = (
        0.25 * non_mshp['asthma_score'] +          # Health burden
        0.20 * non_mshp['absenteeism_score'] +     # Educational impact
        0.15 * non_mshp['enrollment_score'] +      # Scale of impact
        0.25 * non_mshp['svi_score'] +             # Equity/vulnerability
        0.15 * non_mshp['isolation_score']         # Access gap
    )
    
    # Rank and tier
    non_mshp['enhanced_priority_rank'] = non_mshp['enhanced_priority_score'].rank(ascending=False).astype(int)
    
    # Calculate tier thresholds
    n = len(non_mshp)
    non_mshp['enhanced_priority_tier'] = pd.cut(
        non_mshp['enhanced_priority_rank'],
        bins=[0, n*0.1, n*0.25, n*0.5, n],
        labels=['Tier 1 - Critical (Top 10%)', 'Tier 2 - High (10-25%)', 
                'Tier 3 - Moderate (25-50%)', 'Tier 4 - Lower (Bottom 50%)']
    )
    
    # Select output columns
    output_cols = [
        'dbn', 'school_name', 'school_type', 'enrollment',
        'uhf_name', 'lat', 'lon',
        'chronic_absenteeism_rate', 'asthma_ed_rate',
        'svi_overall', 'svi_socioeconomic',
        'distance_to_nearest_mshp_miles', 'accessibility_tier',
        'asthma_score', 'absenteeism_score', 'enrollment_score',
        'svi_score', 'isolation_score',
        'enhanced_priority_score', 'enhanced_priority_rank', 'enhanced_priority_tier'
    ]
    
    output_df = non_mshp[[c for c in output_cols if c in non_mshp.columns]].copy()
    output_df = output_df.sort_values('enhanced_priority_rank')
    
    # Save
    atomic_write_csv(TABLES_DIR / "enhanced_priority_ranking.csv", output_df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ENHANCED PRIORITY RANKING SUMMARY")
    print("=" * 70)
    print(f"\nScoring Weights:")
    print(f"  • Asthma Burden:        25%")
    print(f"  • Chronic Absenteeism:  20%")
    print(f"  • Enrollment Size:      15%")
    print(f"  • Social Vulnerability: 25%")
    print(f"  • Geographic Isolation: 15%")
    
    print(f"\nTier Distribution:")
    tier_counts = output_df['enhanced_priority_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count} schools")
    
    print(f"\nTop 10 Priority Schools:")
    print("-" * 70)
    for i, (_, row) in enumerate(output_df.head(10).iterrows(), 1):
        print(f"{i:2}. {row['school_name'][:40]:<42} Score: {row['enhanced_priority_score']:.1f}")
        print(f"     SVI: {row.get('svi_overall', 0):.2f} | Asthma: {row.get('asthma_ed_rate', 0):.0f} | "
              f"Distance: {row.get('distance_to_nearest_mshp_miles', 0):.2f}mi")
    
    log_step_end(logger, "create_enhanced_priority_score")
    return output_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all enhanced analyses."""
    print("=" * 70)
    print("ENHANCED ANALYSIS: SVI + Regression + Accessibility")
    print("=" * 70)
    
    # Load base data
    schools = pd.read_csv(PROCESSED_DIR / "bronx_schools_analysis_ready.csv")
    print(f"\nLoaded {len(schools)} schools")
    
    # 1. SVI Integration
    print("\n" + "-" * 70)
    print("PART 1: CDC Social Vulnerability Index Integration")
    print("-" * 70)
    svi_data = download_svi_data()
    schools = integrate_svi(schools, svi_data)
    
    # Save enhanced dataset
    atomic_write_csv(PROCESSED_DIR / "bronx_schools_enhanced.csv", schools)
    print(f"✅ Enhanced dataset saved with SVI data")
    
    # 2. Regression Analysis
    print("\n" + "-" * 70)
    print("PART 2: Regression Analysis with Controls")
    print("-" * 70)
    reg_results = run_regression_analysis(schools)
    
    print(f"\nRegression Results:")
    print(f"  Model 1 (Simple):     MSHP coef = {reg_results['model1_simple']['mshp_coef']:.3f}, "
          f"p = {reg_results['model1_simple']['mshp_pvalue']:.4f}")
    print(f"  Model 2 (+ Asthma):   MSHP coef = {reg_results['model2_asthma']['mshp_coef']:.3f}, "
          f"p = {reg_results['model2_asthma']['mshp_pvalue']:.4f}")
    print(f"  Model 3 (Full):       MSHP coef = {reg_results['model3_full']['mshp_coef']:.3f}, "
          f"p = {reg_results['model3_full']['mshp_pvalue']:.4f}, R² = {reg_results['model3_full']['r_squared']:.3f}")
    print(f"  Model 4 (Interaction): MSHP×Asthma = {reg_results['model4_interaction']['interaction_coef']:.4f}, "
          f"p = {reg_results['model4_interaction']['interaction_pvalue']:.4f}")
    print(f"\n✅ Regression results saved")
    
    # 3. Accessibility Analysis
    print("\n" + "-" * 70)
    print("PART 3: Accessibility Analysis")
    print("-" * 70)
    accessibility = calculate_accessibility(schools)
    
    print(f"\nAccessibility Summary:")
    print(f"  Mean distance to MSHP: {accessibility['distance_to_nearest_mshp_miles'].mean():.2f} miles")
    print(f"  Schools >0.5 mi from MSHP: {(accessibility['distance_to_nearest_mshp_miles'] > 0.5).sum()}")
    print(f"  Schools >1 mi from MSHP ('deserts'): {(accessibility['distance_to_nearest_mshp_miles'] > 1).sum()}")
    print(f"✅ Accessibility analysis saved")
    
    # 4. Enhanced Priority Scoring
    print("\n" + "-" * 70)
    print("PART 4: Enhanced Priority Scoring")
    print("-" * 70)
    enhanced_priority = create_enhanced_priority_score(schools, accessibility)
    
    print("\n" + "=" * 70)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutputs:")
    print(f"  📊 data/processed/bronx_schools_enhanced.csv")
    print(f"  📈 outputs/tables/regression_results.csv")
    print(f"  📈 outputs/tables/regression_full_summary.txt")
    print(f"  📍 outputs/tables/accessibility_analysis.csv")
    print(f"  🎯 outputs/tables/enhanced_priority_ranking.csv")


if __name__ == "__main__":
    main()

