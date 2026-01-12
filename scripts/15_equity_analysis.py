#!/usr/bin/env python3
"""
Script 15: Deep Equity Analysis

Input:
    - data/processed/bronx_schools_full.csv

Output:
    - outputs/tables/equity_demographic_disparities.csv
    - outputs/tables/equity_intersectional_risk.csv
    - outputs/tables/equity_by_school_type.csv
    - outputs/tables/high_risk_uncovered_schools.csv

This script performs comprehensive equity analysis across multiple dimensions:
1. Demographic disparity analysis (poverty, race/ethnicity, ELL, SWD)
2. Intersectional risk analysis (high-risk school identification)
3. Coverage by school type analysis
4. Geographic equity (coverage vs health burden by neighborhood)
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import PROCESSED_DIR, TABLES_DIR, CONFIGS_DIR
from mshp_gap.io_utils import atomic_write_csv, write_metadata_sidecar
from mshp_gap.logging_utils import (
    get_logger,
    get_run_id,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("15_equity_analysis")


def load_params():
    """Load parameters from config file."""
    params_file = CONFIGS_DIR / "params.yml"
    if params_file.exists():
        with open(params_file) as f:
            return yaml.safe_load(f)
    return {}


def load_data():
    """Load the full schools dataset."""
    log_step_start(logger, "load_data")
    
    # Try bronx_schools_full.csv first, then analysis_ready
    data_file = PROCESSED_DIR / "bronx_schools_full.csv"
    if not data_file.exists():
        data_file = PROCESSED_DIR / "bronx_schools_analysis_ready.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} schools with {len(df.columns)} columns")
    
    log_step_end(logger, "load_data")
    return df


def analyze_demographic_disparities(df):
    """
    Compare demographics of MSHP vs non-MSHP schools.
    
    Analyzes: poverty, race/ethnicity, ELL, SWD rates
    """
    log_step_start(logger, "analyze_demographic_disparities")
    
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    # Define demographic metrics to analyze
    metrics = {
        'pct_poverty': 'Poverty Rate (%)',
        'pct_black': 'Black Students (%)',
        'pct_hispanic': 'Hispanic Students (%)',
        'pct_white': 'White Students (%)',
        'pct_asian': 'Asian Students (%)',
        'pct_ell': 'English Language Learners (%)',
        'pct_swd': 'Students with Disabilities (%)',
    }
    
    results = []
    
    for col, name in metrics.items():
        if col not in df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue
            
        mshp_vals = mshp[col].dropna() * 100 if mshp[col].max() <= 1 else mshp[col].dropna()
        non_mshp_vals = non_mshp[col].dropna() * 100 if non_mshp[col].max() <= 1 else non_mshp[col].dropna()
        all_vals = df[col].dropna() * 100 if df[col].max() <= 1 else df[col].dropna()
        
        if len(mshp_vals) < 2 or len(non_mshp_vals) < 2:
            continue
        
        # Calculate statistics
        mshp_mean = mshp_vals.mean()
        non_mshp_mean = non_mshp_vals.mean()
        diff = mshp_mean - non_mshp_mean
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(mshp_vals, non_mshp_vals)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((mshp_vals.std()**2 + non_mshp_vals.std()**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        # Calculate coverage rates for high/low groups
        median_val = all_vals.median()
        col_adjusted = col
        high_group = df[df[col] * (100 if df[col].max() <= 1 else 1) > median_val]
        low_group = df[df[col] * (100 if df[col].max() <= 1 else 1) <= median_val]
        
        high_coverage = high_group['has_mshp'].mean() * 100 if len(high_group) > 0 else None
        low_coverage = low_group['has_mshp'].mean() * 100 if len(low_group) > 0 else None
        
        results.append({
            'metric': name,
            'mshp_mean': round(mshp_mean, 1),
            'non_mshp_mean': round(non_mshp_mean, 1),
            'overall_mean': round(all_vals.mean(), 1),
            'difference': round(diff, 2),
            'mshp_n': len(mshp_vals),
            'non_mshp_n': len(non_mshp_vals),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_value, 4),
            'cohens_d': round(cohens_d, 3),
            'significant': p_value < 0.05,
            'high_value_coverage_pct': round(high_coverage, 1) if high_coverage else None,
            'low_value_coverage_pct': round(low_coverage, 1) if low_coverage else None,
        })
        
        effect_desc = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        logger.info(f"{name}: MSHP={mshp_mean:.1f}%, Non-MSHP={non_mshp_mean:.1f}%, diff={diff:+.1f}pp, p={p_value:.4f}, d={cohens_d:.2f} ({effect_desc})")
    
    results_df = pd.DataFrame(results)
    
    log_step_end(logger, "analyze_demographic_disparities")
    return results_df


def analyze_intersectional_risk(df):
    """
    Identify schools with multiple risk factors (high-risk profile).
    
    Risk factors:
    - High poverty (>80%)
    - High chronic absenteeism (>35%)
    - High asthma ED rate (above median)
    - High SWD rate (above median)
    """
    log_step_start(logger, "analyze_intersectional_risk")
    
    # Create risk indicators
    df_risk = df.copy()
    
    # Define thresholds
    poverty_threshold = 0.80  # 80% poverty
    absenteeism_threshold = 35.0  # 35% chronic absenteeism
    asthma_median = df['asthma_ed_rate'].median()
    swd_median = df['pct_swd'].median()
    
    # Create binary risk flags
    df_risk['risk_high_poverty'] = df_risk['pct_poverty'] > poverty_threshold
    df_risk['risk_high_absenteeism'] = df_risk['chronic_absenteeism_rate'] > absenteeism_threshold
    df_risk['risk_high_asthma'] = df_risk['asthma_ed_rate'] > asthma_median
    df_risk['risk_high_swd'] = df_risk['pct_swd'] > swd_median
    
    # Count risk factors per school
    risk_cols = ['risk_high_poverty', 'risk_high_absenteeism', 'risk_high_asthma', 'risk_high_swd']
    df_risk['risk_factor_count'] = df_risk[risk_cols].sum(axis=1)
    
    # Define high-risk as 3+ risk factors
    df_risk['is_high_risk'] = df_risk['risk_factor_count'] >= 3
    
    # Analyze coverage by risk level
    risk_summary = []
    
    for n_risks in range(5):
        subset = df_risk[df_risk['risk_factor_count'] == n_risks]
        if len(subset) == 0:
            continue
            
        mshp_count = subset['has_mshp'].sum()
        total = len(subset)
        coverage_pct = (mshp_count / total) * 100 if total > 0 else 0
        
        risk_summary.append({
            'risk_factor_count': n_risks,
            'total_schools': total,
            'mshp_covered': int(mshp_count),
            'not_covered': total - int(mshp_count),
            'coverage_pct': round(coverage_pct, 1),
            'description': f"{n_risks} risk factors" if n_risks > 0 else "No risk factors",
        })
    
    # Add summary for high-risk (3+)
    high_risk = df_risk[df_risk['is_high_risk'] == True]
    if len(high_risk) > 0:
        mshp_count = high_risk['has_mshp'].sum()
        total = len(high_risk)
        risk_summary.append({
            'risk_factor_count': '3+ (High Risk)',
            'total_schools': total,
            'mshp_covered': int(mshp_count),
            'not_covered': total - int(mshp_count),
            'coverage_pct': round((mshp_count / total) * 100, 1),
            'description': "High-risk schools (3+ factors)",
        })
    
    risk_summary_df = pd.DataFrame(risk_summary)
    
    # Log key finding
    if len(high_risk) > 0:
        uncovered_high_risk = high_risk[high_risk['has_mshp'] == False]
        logger.info(f"High-risk schools: {len(high_risk)} total, {len(uncovered_high_risk)} without MSHP ({len(uncovered_high_risk)/len(high_risk)*100:.1f}%)")
    
    log_step_end(logger, "analyze_intersectional_risk")
    return risk_summary_df, df_risk


def identify_high_risk_uncovered(df_risk):
    """
    Create list of high-risk schools without MSHP coverage.
    """
    log_step_start(logger, "identify_high_risk_uncovered")
    
    # Filter to high-risk uncovered schools
    uncovered = df_risk[(df_risk['is_high_risk'] == True) & (df_risk['has_mshp'] == False)].copy()
    
    # Select relevant columns
    output_cols = [
        'dbn', 'school_name', 'school_type', 'address', 'uhf_name',
        'pct_poverty', 'chronic_absenteeism_rate', 'asthma_ed_rate', 'pct_swd',
        'risk_factor_count', 'enrollment'
    ]
    
    available_cols = [c for c in output_cols if c in uncovered.columns]
    uncovered = uncovered[available_cols].copy()
    
    # Sort by risk factor count and poverty
    uncovered = uncovered.sort_values(
        ['risk_factor_count', 'pct_poverty'], 
        ascending=[False, False]
    )
    
    # Format percentages
    for col in ['pct_poverty', 'pct_swd']:
        if col in uncovered.columns:
            uncovered[col] = (uncovered[col] * 100).round(1)
    
    if 'chronic_absenteeism_rate' in uncovered.columns:
        uncovered['chronic_absenteeism_rate'] = uncovered['chronic_absenteeism_rate'].round(1)
    
    logger.info(f"Identified {len(uncovered)} high-risk uncovered schools")
    
    log_step_end(logger, "identify_high_risk_uncovered")
    return uncovered


def analyze_by_school_type(df):
    """
    Analyze MSHP coverage by school type (elementary, middle, high).
    """
    log_step_start(logger, "analyze_by_school_type")
    
    # Group by school type
    type_summary = df.groupby('school_type').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'chronic_absenteeism_rate': 'mean',
        'enrollment': 'sum',
        'pct_poverty': 'mean',
    }).round(3)
    
    # Flatten column names
    type_summary.columns = [
        'total_schools', 'mshp_count', 'coverage_rate',
        'mean_absenteeism', 'total_enrollment', 'mean_poverty'
    ]
    
    type_summary['coverage_pct'] = (type_summary['coverage_rate'] * 100).round(1)
    type_summary['not_covered'] = type_summary['total_schools'] - type_summary['mshp_count']
    type_summary['mean_poverty_pct'] = (type_summary['mean_poverty'] * 100).round(1)
    type_summary['mean_absenteeism'] = type_summary['mean_absenteeism'].round(1)
    
    # Calculate students served vs not served
    for school_type in type_summary.index:
        subset = df[df['school_type'] == school_type]
        mshp_enrollment = subset[subset['has_mshp'] == True]['enrollment'].sum()
        non_mshp_enrollment = subset[subset['has_mshp'] == False]['enrollment'].sum()
        type_summary.loc[school_type, 'students_covered'] = int(mshp_enrollment)
        type_summary.loc[school_type, 'students_not_covered'] = int(non_mshp_enrollment)
    
    type_summary = type_summary.reset_index()
    
    # Select and order columns
    output_cols = [
        'school_type', 'total_schools', 'mshp_count', 'not_covered', 
        'coverage_pct', 'students_covered', 'students_not_covered',
        'mean_absenteeism', 'mean_poverty_pct'
    ]
    type_summary = type_summary[[c for c in output_cols if c in type_summary.columns]]
    
    # Sort by coverage rate
    type_summary = type_summary.sort_values('coverage_pct', ascending=False)
    
    for _, row in type_summary.iterrows():
        logger.info(f"{row['school_type']}: {row['coverage_pct']:.1f}% coverage ({row['mshp_count']}/{row['total_schools']} schools)")
    
    log_step_end(logger, "analyze_by_school_type")
    return type_summary


def analyze_geographic_equity(df):
    """
    Analyze coverage equity across neighborhoods.
    Identify "doubly disadvantaged" areas (high health burden + low coverage).
    """
    log_step_start(logger, "analyze_geographic_equity")
    
    # Group by neighborhood
    geo_summary = df.groupby('uhf_name').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'asthma_ed_rate': 'first',
        'asthma_hosp_rate': 'first',
        'chronic_absenteeism_rate': 'mean',
        'pct_poverty': 'mean',
        'enrollment': 'sum',
        'childhood_obesity_pct': 'first',
        'youth_mental_health_ed_rate': 'first',
        'health_burden_composite': 'first',
    }).round(3)
    
    # Flatten columns
    geo_summary.columns = [
        'total_schools', 'mshp_count', 'coverage_rate',
        'asthma_ed_rate', 'asthma_hosp_rate',
        'mean_absenteeism', 'mean_poverty', 'total_enrollment',
        'obesity_rate', 'mental_health_rate', 'health_burden'
    ]
    
    geo_summary['coverage_pct'] = (geo_summary['coverage_rate'] * 100).round(1)
    geo_summary['mean_poverty_pct'] = (geo_summary['mean_poverty'] * 100).round(1)
    geo_summary['mean_absenteeism'] = geo_summary['mean_absenteeism'].round(1)
    
    # Calculate health burden percentile (within Bronx)
    geo_summary['health_burden_percentile'] = geo_summary['health_burden'].rank(pct=True) * 100
    
    # Identify quadrants
    coverage_median = geo_summary['coverage_pct'].median()
    health_median = geo_summary['health_burden_percentile'].median()
    
    def classify_quadrant(row):
        high_burden = row['health_burden_percentile'] > health_median
        low_coverage = row['coverage_pct'] < coverage_median
        
        if high_burden and low_coverage:
            return "PRIORITY: High Burden, Low Coverage"
        elif high_burden and not low_coverage:
            return "Good Targeting: High Burden, High Coverage"
        elif not high_burden and low_coverage:
            return "Less Urgent: Low Burden, Low Coverage"
        else:
            return "Over-served?: Low Burden, High Coverage"
    
    geo_summary['equity_quadrant'] = geo_summary.apply(classify_quadrant, axis=1)
    
    geo_summary = geo_summary.reset_index()
    geo_summary = geo_summary.sort_values('health_burden', ascending=False)
    
    # Log key findings
    priority_neighborhoods = geo_summary[geo_summary['equity_quadrant'].str.contains('PRIORITY')]
    if len(priority_neighborhoods) > 0:
        logger.info(f"Priority neighborhoods (high burden, low coverage): {priority_neighborhoods['uhf_name'].tolist()}")
    
    log_step_end(logger, "analyze_geographic_equity")
    return geo_summary


def main():
    """Main entry point for equity analysis."""
    logger.info("=" * 60)
    logger.info("EQUITY ANALYSIS: MSHP COVERAGE DISPARITIES")
    logger.info("=" * 60)
    
    run_id = get_run_id()
    
    # Load data
    df = load_data()
    params = load_params()
    
    # Step 1: Demographic disparity analysis
    logger.info("\n--- Step 1: Demographic Disparity Analysis ---")
    demographic_df = analyze_demographic_disparities(df)
    
    output_path = TABLES_DIR / "equity_demographic_disparities.csv"
    atomic_write_csv(output_path, demographic_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="15_equity_analysis.py",
        run_id=run_id,
        description="Demographic disparities between MSHP and non-MSHP schools",
        inputs=["bronx_schools_full.csv"],
        row_count=len(demographic_df),
        columns=list(demographic_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(demographic_df))
    
    print("\n" + "=" * 60)
    print("DEMOGRAPHIC DISPARITIES: MSHP vs Non-MSHP Schools")
    print("=" * 60)
    display_cols = ['metric', 'mshp_mean', 'non_mshp_mean', 'difference', 'p_value', 'significant']
    print(demographic_df[display_cols].to_string(index=False))
    
    # Step 2: Intersectional risk analysis
    logger.info("\n--- Step 2: Intersectional Risk Analysis ---")
    risk_summary_df, df_with_risk = analyze_intersectional_risk(df)
    
    output_path = TABLES_DIR / "equity_intersectional_risk.csv"
    atomic_write_csv(output_path, risk_summary_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="15_equity_analysis.py",
        run_id=run_id,
        description="MSHP coverage by number of risk factors (poverty, absenteeism, asthma, SWD)",
        inputs=["bronx_schools_full.csv"],
        row_count=len(risk_summary_df),
        columns=list(risk_summary_df.columns),
        thresholds={
            "poverty": ">80%",
            "absenteeism": ">35%",
            "asthma": ">median",
            "swd": ">median"
        }
    )
    log_output_written(logger, output_path, row_count=len(risk_summary_df))
    
    print("\n" + "=" * 60)
    print("MSHP COVERAGE BY RISK LEVEL")
    print("=" * 60)
    print(risk_summary_df.to_string(index=False))
    
    # Step 3: Identify high-risk uncovered schools
    logger.info("\n--- Step 3: Identify High-Risk Uncovered Schools ---")
    high_risk_df = identify_high_risk_uncovered(df_with_risk)
    
    output_path = TABLES_DIR / "high_risk_uncovered_schools.csv"
    atomic_write_csv(output_path, high_risk_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="15_equity_analysis.py",
        run_id=run_id,
        description="High-risk schools (3+ risk factors) without MSHP coverage",
        inputs=["bronx_schools_full.csv"],
        row_count=len(high_risk_df),
        columns=list(high_risk_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(high_risk_df))
    
    print("\n" + "=" * 60)
    print(f"HIGH-RISK SCHOOLS WITHOUT MSHP ({len(high_risk_df)} schools)")
    print("=" * 60)
    if len(high_risk_df) > 0:
        display_cols = ['school_name', 'uhf_name', 'pct_poverty', 'chronic_absenteeism_rate', 'risk_factor_count']
        available_cols = [c for c in display_cols if c in high_risk_df.columns]
        print(high_risk_df.head(15)[available_cols].to_string(index=False))
    
    # Step 4: Coverage by school type
    logger.info("\n--- Step 4: Coverage by School Type ---")
    type_df = analyze_by_school_type(df)
    
    output_path = TABLES_DIR / "equity_by_school_type.csv"
    atomic_write_csv(output_path, type_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="15_equity_analysis.py",
        run_id=run_id,
        description="MSHP coverage rates by school type (elementary, middle, high, etc.)",
        inputs=["bronx_schools_full.csv"],
        row_count=len(type_df),
        columns=list(type_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(type_df))
    
    print("\n" + "=" * 60)
    print("MSHP COVERAGE BY SCHOOL TYPE")
    print("=" * 60)
    print(type_df.to_string(index=False))
    
    # Step 5: Geographic equity analysis
    logger.info("\n--- Step 5: Geographic Equity Analysis ---")
    geo_df = analyze_geographic_equity(df)
    
    output_path = TABLES_DIR / "equity_geographic.csv"
    atomic_write_csv(output_path, geo_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="15_equity_analysis.py",
        run_id=run_id,
        description="Neighborhood-level equity analysis (health burden vs coverage)",
        inputs=["bronx_schools_full.csv"],
        row_count=len(geo_df),
        columns=list(geo_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(geo_df))
    
    print("\n" + "=" * 60)
    print("GEOGRAPHIC EQUITY (Health Burden vs Coverage)")
    print("=" * 60)
    display_cols = ['uhf_name', 'health_burden', 'coverage_pct', 'equity_quadrant']
    available_cols = [c for c in display_cols if c in geo_df.columns]
    print(geo_df[available_cols].to_string(index=False))
    
    # Summary
    print("\n" + "=" * 60)
    print("EQUITY ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Key findings
    print("\n📊 KEY FINDINGS:")
    
    # Find the biggest disparity
    if len(demographic_df) > 0:
        sig_disparities = demographic_df[demographic_df['significant'] == True]
        if len(sig_disparities) > 0:
            biggest = sig_disparities.iloc[sig_disparities['cohens_d'].abs().argmax()]
            direction = "higher" if biggest['difference'] > 0 else "lower"
            print(f"  • MSHP schools have {abs(biggest['difference']):.1f}pp {direction} {biggest['metric'].lower()}")
    
    # High-risk finding
    if len(high_risk_df) > 0:
        total_high_risk = len(df_with_risk[df_with_risk['is_high_risk'] == True])
        pct_uncovered = len(high_risk_df) / total_high_risk * 100 if total_high_risk > 0 else 0
        print(f"  • {len(high_risk_df)} high-risk schools ({pct_uncovered:.0f}%) lack MSHP coverage")
    
    # School type finding
    if len(type_df) > 0:
        min_coverage = type_df[type_df['coverage_pct'] == type_df['coverage_pct'].min()].iloc[0]
        print(f"  • {min_coverage['school_type']} schools have lowest coverage ({min_coverage['coverage_pct']:.1f}%)")
    
    print(f"\nOutputs saved to: {TABLES_DIR}")
    print("  - equity_demographic_disparities.csv")
    print("  - equity_intersectional_risk.csv")
    print("  - high_risk_uncovered_schools.csv")
    print("  - equity_by_school_type.csv")
    print("  - equity_geographic.csv")


if __name__ == "__main__":
    main()

