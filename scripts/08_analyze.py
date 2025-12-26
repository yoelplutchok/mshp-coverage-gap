#!/usr/bin/env python3
"""
Script 08: Analysis & Priority Ranking

Input:
    - data/processed/bronx_schools_analysis_ready.csv

Output:
    - outputs/tables/summary_statistics.csv
    - outputs/tables/mshp_vs_non_mshp_comparison.csv
    - outputs/tables/neighborhood_summary.csv
    - outputs/tables/non_mshp_schools_priority_ranked.csv
    - outputs/tables/statistical_tests.csv

This script performs Phase 3 analysis:
1. Calculate summary statistics (MSHP vs non-MSHP)
2. Run statistical tests (t-test for absenteeism comparison)
3. Create priority ranking for non-MSHP schools
4. Generate neighborhood-level aggregations
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

logger = get_logger("08_analyze")


def load_params():
    """Load parameters from config file."""
    params_file = CONFIGS_DIR / "params.yml"
    if params_file.exists():
        with open(params_file) as f:
            return yaml.safe_load(f)
    return {
        "priority_weights": {
            "asthma_burden": 0.4,
            "chronic_absenteeism": 0.4,
            "enrollment_size": 0.2,
        },
        "priority_tiers": {
            "tier_1_critical": 80,
            "tier_2_high": 60,
            "tier_3_moderate": 40,
        }
    }


def load_analysis_data():
    """Load the analysis-ready dataset."""
    log_step_start(logger, "load_analysis_data")
    
    data_file = PROCESSED_DIR / "bronx_schools_analysis_ready.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Analysis file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} schools for analysis")
    
    log_step_end(logger, "load_analysis_data")
    return df


def calculate_summary_statistics(df):
    """Calculate overall summary statistics."""
    log_step_start(logger, "calculate_summary_statistics")
    
    summary = {
        "metric": [],
        "all_schools": [],
        "mshp_covered": [],
        "not_covered": [],
    }
    
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    # School counts
    summary["metric"].append("Total Schools")
    summary["all_schools"].append(len(df))
    summary["mshp_covered"].append(len(mshp))
    summary["not_covered"].append(len(non_mshp))
    
    # Chronic absenteeism
    for stat_name, stat_func in [
        ("Chronic Absenteeism - Mean", "mean"),
        ("Chronic Absenteeism - Median", "median"),
        ("Chronic Absenteeism - Std Dev", "std"),
    ]:
        summary["metric"].append(stat_name)
        all_vals = df['chronic_absenteeism_rate'].dropna()
        mshp_vals = mshp['chronic_absenteeism_rate'].dropna()
        non_mshp_vals = non_mshp['chronic_absenteeism_rate'].dropna()
        
        if stat_func == "mean":
            summary["all_schools"].append(round(all_vals.mean(), 2) if len(all_vals) > 0 else None)
            summary["mshp_covered"].append(round(mshp_vals.mean(), 2) if len(mshp_vals) > 0 else None)
            summary["not_covered"].append(round(non_mshp_vals.mean(), 2) if len(non_mshp_vals) > 0 else None)
        elif stat_func == "median":
            summary["all_schools"].append(round(all_vals.median(), 2) if len(all_vals) > 0 else None)
            summary["mshp_covered"].append(round(mshp_vals.median(), 2) if len(mshp_vals) > 0 else None)
            summary["not_covered"].append(round(non_mshp_vals.median(), 2) if len(non_mshp_vals) > 0 else None)
        else:
            summary["all_schools"].append(round(all_vals.std(), 2) if len(all_vals) > 0 else None)
            summary["mshp_covered"].append(round(mshp_vals.std(), 2) if len(mshp_vals) > 0 else None)
            summary["not_covered"].append(round(non_mshp_vals.std(), 2) if len(non_mshp_vals) > 0 else None)
    
    # Enrollment
    for stat_name, stat_func in [
        ("Enrollment - Mean", "mean"),
        ("Enrollment - Median", "median"),
    ]:
        summary["metric"].append(stat_name)
        all_vals = df['enrollment'].dropna()
        mshp_vals = mshp['enrollment'].dropna()
        non_mshp_vals = non_mshp['enrollment'].dropna()
        
        if stat_func == "mean":
            summary["all_schools"].append(round(all_vals.mean(), 0) if len(all_vals) > 0 else None)
            summary["mshp_covered"].append(round(mshp_vals.mean(), 0) if len(mshp_vals) > 0 else None)
            summary["not_covered"].append(round(non_mshp_vals.mean(), 0) if len(non_mshp_vals) > 0 else None)
        else:
            summary["all_schools"].append(round(all_vals.median(), 0) if len(all_vals) > 0 else None)
            summary["mshp_covered"].append(round(mshp_vals.median(), 0) if len(mshp_vals) > 0 else None)
            summary["not_covered"].append(round(non_mshp_vals.median(), 0) if len(non_mshp_vals) > 0 else None)
    
    summary_df = pd.DataFrame(summary)
    
    log_step_end(logger, "calculate_summary_statistics")
    return summary_df


def run_statistical_tests(df):
    """Run statistical tests comparing MSHP vs non-MSHP schools."""
    log_step_start(logger, "run_statistical_tests")
    
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    tests = []
    
    # T-test for chronic absenteeism
    mshp_absence = mshp['chronic_absenteeism_rate'].dropna()
    non_mshp_absence = non_mshp['chronic_absenteeism_rate'].dropna()
    
    if len(mshp_absence) > 1 and len(non_mshp_absence) > 1:
        t_stat, p_value = stats.ttest_ind(mshp_absence, non_mshp_absence)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((mshp_absence.std()**2 + non_mshp_absence.std()**2) / 2)
        cohens_d = (mshp_absence.mean() - non_mshp_absence.mean()) / pooled_std if pooled_std > 0 else 0
        
        tests.append({
            "test": "Independent t-test: Chronic Absenteeism",
            "group_1": "MSHP Covered",
            "group_1_n": len(mshp_absence),
            "group_1_mean": round(mshp_absence.mean(), 2),
            "group_2": "Not Covered",
            "group_2_n": len(non_mshp_absence),
            "group_2_mean": round(non_mshp_absence.mean(), 2),
            "difference": round(non_mshp_absence.mean() - mshp_absence.mean(), 2),
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "cohens_d": round(cohens_d, 3),
            "significant_at_05": p_value < 0.05,
        })
        
        logger.info(f"T-test p-value: {p_value:.4f} (significant: {p_value < 0.05})")
    
    # Mann-Whitney U test (non-parametric alternative)
    if len(mshp_absence) > 1 and len(non_mshp_absence) > 1:
        u_stat, p_value_mw = stats.mannwhitneyu(mshp_absence, non_mshp_absence, alternative='two-sided')
        
        tests.append({
            "test": "Mann-Whitney U: Chronic Absenteeism",
            "group_1": "MSHP Covered",
            "group_1_n": len(mshp_absence),
            "group_1_mean": round(mshp_absence.mean(), 2),
            "group_2": "Not Covered",
            "group_2_n": len(non_mshp_absence),
            "group_2_mean": round(non_mshp_absence.mean(), 2),
            "difference": round(non_mshp_absence.mean() - mshp_absence.mean(), 2),
            "t_statistic": round(u_stat, 3),
            "p_value": round(p_value_mw, 4),
            "cohens_d": None,
            "significant_at_05": p_value_mw < 0.05,
        })
    
    # Correlation: Asthma rate vs MSHP coverage (at school level)
    df_with_asthma = df[df['asthma_ed_rate'].notna() & df['has_mshp'].notna()]
    if len(df_with_asthma) > 2:
        # Point-biserial correlation (binary vs continuous)
        r, p_corr = stats.pointbiserialr(df_with_asthma['has_mshp'].astype(int), df_with_asthma['asthma_ed_rate'])
        
        tests.append({
            "test": "Point-biserial Correlation: MSHP Coverage vs Asthma Rate",
            "group_1": "MSHP (binary)",
            "group_1_n": len(df_with_asthma),
            "group_1_mean": None,
            "group_2": "Asthma ED Rate",
            "group_2_n": len(df_with_asthma),
            "group_2_mean": round(df_with_asthma['asthma_ed_rate'].mean(), 2),
            "difference": None,
            "t_statistic": round(r, 3),
            "p_value": round(p_corr, 4),
            "cohens_d": None,
            "significant_at_05": p_corr < 0.05,
        })
        
        logger.info(f"Correlation (MSHP vs Asthma): r={r:.3f}, p={p_corr:.4f}")
    
    tests_df = pd.DataFrame(tests)
    
    log_step_end(logger, "run_statistical_tests")
    return tests_df


def calculate_neighborhood_summary(df):
    """Calculate neighborhood-level summary statistics."""
    log_step_start(logger, "calculate_neighborhood_summary")
    
    # Group by UHF neighborhood
    neighborhood_stats = df.groupby('uhf_name').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'chronic_absenteeism_rate': ['mean', 'median'],
        'enrollment': 'sum',
        'asthma_ed_rate': 'first',
        'asthma_hosp_rate': 'first',
        'uhf_code': 'first',
    }).round(2)
    
    # Flatten column names
    neighborhood_stats.columns = [
        'total_schools', 'mshp_count', 'mshp_coverage_pct',
        'mean_absenteeism', 'median_absenteeism',
        'total_enrollment', 'asthma_ed_rate', 'asthma_hosp_rate', 'uhf_code'
    ]
    
    neighborhood_stats['mshp_coverage_pct'] = (neighborhood_stats['mshp_coverage_pct'] * 100).round(1)
    
    # Calculate coverage gap score (high asthma + low coverage = high gap)
    # Normalize asthma rate to 0-100 scale
    asthma_min = neighborhood_stats['asthma_ed_rate'].min()
    asthma_max = neighborhood_stats['asthma_ed_rate'].max()
    neighborhood_stats['asthma_percentile'] = (
        (neighborhood_stats['asthma_ed_rate'] - asthma_min) / (asthma_max - asthma_min) * 100
    ).round(1)
    
    # Coverage gap = asthma burden × (1 - coverage rate)
    neighborhood_stats['coverage_gap_score'] = (
        neighborhood_stats['asthma_percentile'] * (100 - neighborhood_stats['mshp_coverage_pct']) / 100
    ).round(1)
    
    # Sort by coverage gap score (highest need first)
    neighborhood_stats = neighborhood_stats.sort_values('coverage_gap_score', ascending=False)
    
    # Reset index to make uhf_name a column
    neighborhood_stats = neighborhood_stats.reset_index()
    
    logger.info(f"Calculated stats for {len(neighborhood_stats)} neighborhoods")
    
    log_step_end(logger, "calculate_neighborhood_summary")
    return neighborhood_stats


def create_priority_ranking(df, params):
    """Create priority ranking for non-MSHP schools."""
    log_step_start(logger, "create_priority_ranking")
    
    # Filter to non-MSHP schools only
    non_mshp = df[df['has_mshp'] == False].copy()
    logger.info(f"Ranking {len(non_mshp)} non-MSHP schools")
    
    # Get weights from params
    weights = params.get("priority_weights", {
        "asthma_burden": 0.4,
        "chronic_absenteeism": 0.4,
        "enrollment_size": 0.2,
    })
    
    tiers = params.get("priority_tiers", {
        "tier_1_critical": 80,
        "tier_2_high": 60,
        "tier_3_moderate": 40,
    })
    
    # Normalize each component to 0-100 scale using percentile ranks
    
    # Asthma burden (higher = higher priority)
    non_mshp['asthma_score'] = non_mshp['asthma_ed_rate'].rank(pct=True) * 100
    
    # Chronic absenteeism (higher = higher priority)
    # Handle missing values by using median
    absence_filled = non_mshp['chronic_absenteeism_rate'].fillna(
        non_mshp['chronic_absenteeism_rate'].median()
    )
    non_mshp['absenteeism_score'] = absence_filled.rank(pct=True) * 100
    
    # Enrollment (higher = higher priority - more students helped)
    enrollment_filled = non_mshp['enrollment'].fillna(non_mshp['enrollment'].median())
    non_mshp['enrollment_score'] = enrollment_filled.rank(pct=True) * 100
    
    # Calculate weighted priority score
    non_mshp['priority_score'] = (
        weights['asthma_burden'] * non_mshp['asthma_score'] +
        weights['chronic_absenteeism'] * non_mshp['absenteeism_score'] +
        weights['enrollment_size'] * non_mshp['enrollment_score']
    ).round(2)
    
    # Assign tiers based on percentile thresholds
    def assign_tier(score, percentile):
        if percentile >= tiers['tier_1_critical']:
            return "Tier 1 - Critical"
        elif percentile >= tiers['tier_2_high']:
            return "Tier 2 - High"
        elif percentile >= tiers['tier_3_moderate']:
            return "Tier 3 - Moderate"
        else:
            return "Tier 4 - Lower"
    
    # Calculate percentile for each school
    non_mshp['priority_percentile'] = non_mshp['priority_score'].rank(pct=True) * 100
    non_mshp['priority_tier'] = non_mshp.apply(
        lambda row: assign_tier(row['priority_score'], row['priority_percentile']), axis=1
    )
    
    # Sort by priority score (highest first)
    non_mshp = non_mshp.sort_values('priority_score', ascending=False)
    
    # Add rank
    non_mshp['priority_rank'] = range(1, len(non_mshp) + 1)
    
    # Select columns for output
    output_cols = [
        'priority_rank', 'dbn', 'school_name', 'school_type', 'address',
        'uhf_name', 'asthma_ed_rate', 'chronic_absenteeism_rate', 'enrollment',
        'asthma_score', 'absenteeism_score', 'enrollment_score',
        'priority_score', 'priority_tier'
    ]
    
    result = non_mshp[[c for c in output_cols if c in non_mshp.columns]].copy()
    
    # Round scores
    for col in ['asthma_score', 'absenteeism_score', 'enrollment_score', 'priority_score']:
        if col in result.columns:
            result[col] = result[col].round(1)
    
    # Log tier distribution
    tier_counts = result['priority_tier'].value_counts()
    logger.info(f"Priority tier distribution:\n{tier_counts.to_string()}")
    
    log_step_end(logger, "create_priority_ranking")
    return result


def main():
    """Main entry point for Phase 3 analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ANALYSIS & PRIORITY RANKING")
    logger.info("=" * 60)
    
    run_id = get_run_id()
    
    # Load data and params
    df = load_analysis_data()
    params = load_params()
    
    # Step 1: Summary statistics
    logger.info("\n--- Step 1: Calculate summary statistics ---")
    summary_df = calculate_summary_statistics(df)
    
    output_path = TABLES_DIR / "summary_statistics.csv"
    atomic_write_csv(output_path, summary_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="08_analyze.py",
        run_id=run_id,
        description="Summary statistics comparing all schools, MSHP-covered, and non-covered",
        inputs=["bronx_schools_analysis_ready.csv"],
        row_count=len(summary_df),
        columns=list(summary_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(summary_df))
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    
    # Step 2: Statistical tests
    logger.info("\n--- Step 2: Run statistical tests ---")
    tests_df = run_statistical_tests(df)
    
    output_path = TABLES_DIR / "statistical_tests.csv"
    atomic_write_csv(output_path, tests_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="08_analyze.py",
        run_id=run_id,
        description="Statistical tests comparing MSHP vs non-MSHP schools",
        inputs=["bronx_schools_analysis_ready.csv"],
        row_count=len(tests_df),
        columns=list(tests_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(tests_df))
    
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    for _, row in tests_df.iterrows():
        print(f"\n{row['test']}")
        print(f"  {row['group_1']}: n={row['group_1_n']}, mean={row['group_1_mean']}")
        print(f"  {row['group_2']}: n={row['group_2_n']}, mean={row['group_2_mean']}")
        if row['difference'] is not None:
            print(f"  Difference: {row['difference']}")
        print(f"  p-value: {row['p_value']} {'✓ SIGNIFICANT' if row['significant_at_05'] else '(not significant)'}")
    
    # Step 3: Neighborhood summary
    logger.info("\n--- Step 3: Neighborhood-level summary ---")
    neighborhood_df = calculate_neighborhood_summary(df)
    
    output_path = TABLES_DIR / "neighborhood_summary.csv"
    atomic_write_csv(output_path, neighborhood_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="08_analyze.py",
        run_id=run_id,
        description="Neighborhood-level summary with coverage gap scores",
        inputs=["bronx_schools_analysis_ready.csv"],
        row_count=len(neighborhood_df),
        columns=list(neighborhood_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(neighborhood_df))
    
    print("\n" + "=" * 60)
    print("NEIGHBORHOOD SUMMARY (sorted by coverage gap)")
    print("=" * 60)
    display_cols = ['uhf_name', 'total_schools', 'mshp_count', 'mshp_coverage_pct', 
                    'asthma_ed_rate', 'coverage_gap_score']
    print(neighborhood_df[display_cols].to_string(index=False))
    
    # Step 4: Priority ranking
    logger.info("\n--- Step 4: Create priority ranking for non-MSHP schools ---")
    priority_df = create_priority_ranking(df, params)
    
    output_path = TABLES_DIR / "non_mshp_schools_priority_ranked.csv"
    atomic_write_csv(output_path, priority_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="08_analyze.py",
        run_id=run_id,
        description="Non-MSHP schools ranked by expansion priority",
        inputs=["bronx_schools_analysis_ready.csv", "configs/params.yml"],
        row_count=len(priority_df),
        columns=list(priority_df.columns),
        weights=params.get("priority_weights"),
    )
    log_output_written(logger, output_path, row_count=len(priority_df))
    
    print("\n" + "=" * 60)
    print("TOP 20 PRIORITY SCHOOLS FOR MSHP EXPANSION")
    print("=" * 60)
    display_cols = ['priority_rank', 'school_name', 'uhf_name', 
                    'asthma_ed_rate', 'chronic_absenteeism_rate', 'priority_score', 'priority_tier']
    print(priority_df.head(20)[display_cols].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("PRIORITY TIER DISTRIBUTION")
    print("=" * 60)
    tier_counts = priority_df['priority_tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier}: {count} schools")
    
    print("\n" + "=" * 60)
    print("PHASE 3 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {TABLES_DIR}")
    print("  - summary_statistics.csv")
    print("  - statistical_tests.csv")
    print("  - neighborhood_summary.csv")
    print("  - non_mshp_schools_priority_ranked.csv")


if __name__ == "__main__":
    main()

