#!/usr/bin/env python3
"""
Script 16: Health Burden Deep Dive

Input:
    - data/processed/bronx_schools_full.csv

Output:
    - outputs/tables/health_burden_composite_scores.csv
    - outputs/tables/health_coverage_mismatch.csv
    - outputs/tables/neighborhood_health_rankings.csv
    - outputs/tables/condition_specific_summary.csv

This script performs comprehensive health burden analysis:
1. Create weighted health burden composite score
2. Analyze health-coverage mismatch (scatter quadrant analysis)
3. Rank neighborhoods by comprehensive health burden
4. Condition-specific analysis (asthma, obesity, mental health, lead)
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

logger = get_logger("16_health_burden_analysis")


# Health burden weights - prioritize conditions addressable by school health programs
HEALTH_WEIGHTS = {
    'asthma_ed_rate': 0.25,           # Directly addressable by MSHP
    'childhood_obesity_pct': 0.15,     # Addressable through health education
    'youth_mental_health_ed_rate': 0.20,  # Growing need, addressable
    'lead_elevated_pct': 0.10,         # Screening referrals
    'food_insecurity_pct': 0.15,       # Related to health outcomes
    'preventable_hosp_rate': 0.15,     # Indicator of healthcare access
}


def load_data():
    """Load the full schools dataset."""
    log_step_start(logger, "load_data")
    
    data_file = PROCESSED_DIR / "bronx_schools_full.csv"
    if not data_file.exists():
        data_file = PROCESSED_DIR / "bronx_schools_analysis_ready.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} schools with {len(df.columns)} columns")
    
    log_step_end(logger, "load_data")
    return df


def calculate_health_burden_composite(df):
    """
    Calculate weighted health burden composite score for each school/neighborhood.
    
    Uses multiple health indicators weighted by relevance to school health programs.
    """
    log_step_start(logger, "calculate_health_burden_composite")
    
    df_health = df.copy()
    
    # Identify available health columns
    available_cols = [col for col in HEALTH_WEIGHTS.keys() if col in df.columns]
    logger.info(f"Available health indicators: {available_cols}")
    
    # Normalize each health indicator to 0-100 scale (percentile rank)
    for col in available_cols:
        col_normalized = f"{col}_normalized"
        df_health[col_normalized] = df_health[col].rank(pct=True, na_option='keep') * 100
    
    # Calculate weighted composite
    total_weight = sum(HEALTH_WEIGHTS[col] for col in available_cols)
    
    df_health['health_burden_weighted'] = 0
    for col in available_cols:
        weight = HEALTH_WEIGHTS[col] / total_weight  # Renormalize weights
        col_normalized = f"{col}_normalized"
        df_health['health_burden_weighted'] += weight * df_health[col_normalized].fillna(50)  # Use median for missing
    
    df_health['health_burden_weighted'] = df_health['health_burden_weighted'].round(2)
    
    # Rank schools by composite burden
    df_health['health_burden_rank'] = df_health['health_burden_weighted'].rank(ascending=False, method='dense').astype(int)
    
    # Assign burden tiers
    def assign_burden_tier(percentile):
        if percentile >= 80:
            return "Very High"
        elif percentile >= 60:
            return "High"
        elif percentile >= 40:
            return "Moderate"
        elif percentile >= 20:
            return "Low"
        else:
            return "Very Low"
    
    df_health['burden_percentile'] = df_health['health_burden_weighted'].rank(pct=True) * 100
    df_health['burden_tier'] = df_health['burden_percentile'].apply(assign_burden_tier)
    
    # Log summary
    tier_counts = df_health['burden_tier'].value_counts()
    logger.info(f"Burden tier distribution:\n{tier_counts.to_string()}")
    
    log_step_end(logger, "calculate_health_burden_composite")
    return df_health


def analyze_health_coverage_mismatch(df_health):
    """
    Analyze mismatch between health burden and MSHP coverage.
    
    Creates quadrant analysis:
    - High burden, low coverage = PRIORITY
    - High burden, high coverage = Good targeting
    - Low burden, low coverage = Less urgent
    - Low burden, high coverage = Potentially over-served
    """
    log_step_start(logger, "analyze_health_coverage_mismatch")
    
    # Aggregate to neighborhood level
    neighborhood = df_health.groupby('uhf_name').agg({
        'dbn': 'count',
        'has_mshp': ['sum', 'mean'],
        'health_burden_weighted': 'mean',
        'asthma_ed_rate': 'first',
        'childhood_obesity_pct': 'first',
        'youth_mental_health_ed_rate': 'first',
        'enrollment': 'sum',
        'chronic_absenteeism_rate': 'mean',
    }).round(2)
    
    neighborhood.columns = [
        'total_schools', 'mshp_count', 'coverage_rate',
        'health_burden', 'asthma_rate', 'obesity_rate', 'mental_health_rate',
        'total_enrollment', 'mean_absenteeism'
    ]
    
    neighborhood['coverage_pct'] = (neighborhood['coverage_rate'] * 100).round(1)
    neighborhood = neighborhood.reset_index()
    
    # Calculate percentiles for burden and coverage
    neighborhood['burden_percentile'] = neighborhood['health_burden'].rank(pct=True) * 100
    neighborhood['coverage_percentile'] = neighborhood['coverage_pct'].rank(pct=True) * 100
    
    # Classify into quadrants (using median as threshold)
    burden_median = neighborhood['burden_percentile'].median()
    coverage_median = neighborhood['coverage_percentile'].median()
    
    def classify_mismatch(row):
        high_burden = row['burden_percentile'] >= 50  # Above median
        high_coverage = row['coverage_percentile'] >= 50  # Above median
        
        if high_burden and not high_coverage:
            return "HIGH PRIORITY: High Burden, Low Coverage"
        elif high_burden and high_coverage:
            return "Good: High Burden, High Coverage"
        elif not high_burden and not high_coverage:
            return "Monitor: Low Burden, Low Coverage"
        else:
            return "Review: Low Burden, High Coverage"
    
    neighborhood['mismatch_quadrant'] = neighborhood.apply(classify_mismatch, axis=1)
    
    # Calculate mismatch score (higher = more mismatch = more priority)
    # Score = burden_percentile - coverage_percentile (positive = underserved)
    neighborhood['mismatch_score'] = (
        neighborhood['burden_percentile'] - neighborhood['coverage_percentile']
    ).round(1)
    
    # Sort by mismatch score (highest priority first)
    neighborhood = neighborhood.sort_values('mismatch_score', ascending=False)
    
    # Log findings
    priority = neighborhood[neighborhood['mismatch_quadrant'].str.contains('PRIORITY')]
    if len(priority) > 0:
        logger.info(f"High-priority neighborhoods: {priority['uhf_name'].tolist()}")
    
    log_step_end(logger, "analyze_health_coverage_mismatch")
    return neighborhood


def rank_neighborhoods_by_health(df_health):
    """
    Create comprehensive neighborhood health rankings.
    """
    log_step_start(logger, "rank_neighborhoods_by_health")
    
    # Get available health columns
    health_cols = [col for col in HEALTH_WEIGHTS.keys() if col in df_health.columns]
    
    # Aggregate to neighborhood
    agg_dict = {
        'dbn': 'count',
        'has_mshp': 'mean',
        'health_burden_weighted': 'mean',
        'enrollment': 'sum',
        'chronic_absenteeism_rate': 'mean',
        'pct_poverty': 'mean',
    }
    
    # Add health columns
    for col in health_cols:
        agg_dict[col] = 'first'  # These are already at neighborhood level
    
    neighborhood = df_health.groupby('uhf_name').agg(agg_dict).round(3)
    
    # Rename columns
    neighborhood = neighborhood.rename(columns={
        'dbn': 'total_schools',
        'has_mshp': 'mshp_coverage_rate',
        'health_burden_weighted': 'composite_burden_score',
        'enrollment': 'total_students',
        'chronic_absenteeism_rate': 'avg_absenteeism',
        'pct_poverty': 'avg_poverty_rate',
    })
    
    neighborhood['mshp_coverage_pct'] = (neighborhood['mshp_coverage_rate'] * 100).round(1)
    neighborhood['avg_poverty_pct'] = (neighborhood['avg_poverty_rate'] * 100).round(1)
    neighborhood['avg_absenteeism'] = neighborhood['avg_absenteeism'].round(1)
    
    # Rank by composite burden
    neighborhood['burden_rank'] = neighborhood['composite_burden_score'].rank(ascending=False, method='dense').astype(int)
    
    neighborhood = neighborhood.reset_index()
    neighborhood = neighborhood.sort_values('burden_rank')
    
    log_step_end(logger, "rank_neighborhoods_by_health")
    return neighborhood


def analyze_condition_specific(df_health):
    """
    Analyze MSHP coverage relative to each specific health condition.
    """
    log_step_start(logger, "analyze_condition_specific")
    
    conditions = {
        'asthma_ed_rate': 'Asthma ED Visits (per 10K)',
        'childhood_obesity_pct': 'Childhood Obesity (%)',
        'youth_mental_health_ed_rate': 'Youth Mental Health ED (per 10K)',
        'lead_elevated_pct': 'Elevated Blood Lead (%)',
        'food_insecurity_pct': 'Food Insecurity (%)',
        'preventable_hosp_rate': 'Preventable Hospitalizations (per 10K)',
    }
    
    results = []
    
    for col, name in conditions.items():
        if col not in df_health.columns:
            continue
            
        # Split into high vs low burden groups (median split)
        valid_data = df_health[df_health[col].notna()]
        median_val = valid_data[col].median()
        
        high_burden = valid_data[valid_data[col] > median_val]
        low_burden = valid_data[valid_data[col] <= median_val]
        
        # Calculate coverage rates
        high_coverage = high_burden['has_mshp'].mean() * 100
        low_coverage = low_burden['has_mshp'].mean() * 100
        coverage_gap = high_coverage - low_coverage
        
        # Statistical test for difference
        if len(high_burden) > 0 and len(low_burden) > 0:
            # Chi-square test for coverage difference
            contingency = pd.crosstab(
                valid_data[col] > median_val,
                valid_data['has_mshp']
            )
            if contingency.shape == (2, 2):
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            else:
                chi2, p_value = 0, 1.0
        else:
            chi2, p_value = 0, 1.0
        
        results.append({
            'condition': name,
            'metric_column': col,
            'median_value': round(median_val, 2),
            'high_burden_schools': len(high_burden),
            'high_burden_coverage_pct': round(high_coverage, 1),
            'low_burden_schools': len(low_burden),
            'low_burden_coverage_pct': round(low_coverage, 1),
            'coverage_gap_pp': round(coverage_gap, 1),
            'chi_squared': round(chi2, 2),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
            'targeting_direction': 'Toward high burden' if coverage_gap > 0 else 'Away from high burden',
        })
        
        logger.info(f"{name}: High burden coverage={high_coverage:.1f}%, Low={low_coverage:.1f}%, gap={coverage_gap:+.1f}pp")
    
    results_df = pd.DataFrame(results)
    
    log_step_end(logger, "analyze_condition_specific")
    return results_df


def main():
    """Main entry point for health burden analysis."""
    logger.info("=" * 60)
    logger.info("HEALTH BURDEN DEEP DIVE ANALYSIS")
    logger.info("=" * 60)
    
    run_id = get_run_id()
    
    # Load data
    df = load_data()
    
    # Step 1: Calculate health burden composite scores
    logger.info("\n--- Step 1: Calculate Health Burden Composite ---")
    df_health = calculate_health_burden_composite(df)
    
    # Save school-level health burden scores
    output_cols = [
        'dbn', 'school_name', 'uhf_name', 'has_mshp',
        'health_burden_weighted', 'health_burden_rank', 'burden_tier',
        'asthma_ed_rate', 'childhood_obesity_pct', 'youth_mental_health_ed_rate',
        'lead_elevated_pct', 'food_insecurity_pct', 'enrollment'
    ]
    available_cols = [c for c in output_cols if c in df_health.columns]
    school_burden = df_health[available_cols].sort_values('health_burden_rank')
    
    output_path = TABLES_DIR / "health_burden_composite_scores.csv"
    atomic_write_csv(output_path, school_burden)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="16_health_burden_analysis.py",
        run_id=run_id,
        description="Weighted health burden composite scores for each school",
        inputs=["bronx_schools_full.csv"],
        row_count=len(school_burden),
        columns=list(school_burden.columns),
        weights=HEALTH_WEIGHTS,
    )
    log_output_written(logger, output_path, row_count=len(school_burden))
    
    print("\n" + "=" * 60)
    print("SCHOOLS BY HEALTH BURDEN (Top 20)")
    print("=" * 60)
    display_cols = ['school_name', 'uhf_name', 'has_mshp', 'health_burden_weighted', 'burden_tier']
    available_display = [c for c in display_cols if c in school_burden.columns]
    print(school_burden.head(20)[available_display].to_string(index=False))
    
    # Step 2: Health-coverage mismatch analysis
    logger.info("\n--- Step 2: Health-Coverage Mismatch Analysis ---")
    mismatch_df = analyze_health_coverage_mismatch(df_health)
    
    output_path = TABLES_DIR / "health_coverage_mismatch.csv"
    atomic_write_csv(output_path, mismatch_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="16_health_burden_analysis.py",
        run_id=run_id,
        description="Neighborhood-level health burden vs MSHP coverage mismatch analysis",
        inputs=["bronx_schools_full.csv"],
        row_count=len(mismatch_df),
        columns=list(mismatch_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(mismatch_df))
    
    print("\n" + "=" * 60)
    print("HEALTH-COVERAGE MISMATCH BY NEIGHBORHOOD")
    print("=" * 60)
    display_cols = ['uhf_name', 'health_burden', 'coverage_pct', 'mismatch_score', 'mismatch_quadrant']
    print(mismatch_df[display_cols].to_string(index=False))
    
    # Step 3: Neighborhood health rankings
    logger.info("\n--- Step 3: Neighborhood Health Rankings ---")
    rankings_df = rank_neighborhoods_by_health(df_health)
    
    output_path = TABLES_DIR / "neighborhood_health_rankings.csv"
    atomic_write_csv(output_path, rankings_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="16_health_burden_analysis.py",
        run_id=run_id,
        description="Comprehensive neighborhood health rankings",
        inputs=["bronx_schools_full.csv"],
        row_count=len(rankings_df),
        columns=list(rankings_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(rankings_df))
    
    print("\n" + "=" * 60)
    print("NEIGHBORHOOD HEALTH RANKINGS")
    print("=" * 60)
    display_cols = ['burden_rank', 'uhf_name', 'composite_burden_score', 'mshp_coverage_pct', 'total_students']
    available_display = [c for c in display_cols if c in rankings_df.columns]
    print(rankings_df[available_display].to_string(index=False))
    
    # Step 4: Condition-specific analysis
    logger.info("\n--- Step 4: Condition-Specific Analysis ---")
    condition_df = analyze_condition_specific(df_health)
    
    output_path = TABLES_DIR / "condition_specific_summary.csv"
    atomic_write_csv(output_path, condition_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="16_health_burden_analysis.py",
        run_id=run_id,
        description="MSHP coverage analysis by specific health condition",
        inputs=["bronx_schools_full.csv"],
        row_count=len(condition_df),
        columns=list(condition_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(condition_df))
    
    print("\n" + "=" * 60)
    print("MSHP TARGETING BY HEALTH CONDITION")
    print("=" * 60)
    display_cols = ['condition', 'high_burden_coverage_pct', 'low_burden_coverage_pct', 'coverage_gap_pp', 'significant']
    print(condition_df[display_cols].to_string(index=False))
    
    # Summary
    print("\n" + "=" * 60)
    print("HEALTH BURDEN ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\n📊 KEY FINDINGS:")
    
    # Priority neighborhoods
    priority = mismatch_df[mismatch_df['mismatch_quadrant'].str.contains('PRIORITY')]
    if len(priority) > 0:
        print(f"  • {len(priority)} neighborhoods are high-priority (high burden, low coverage):")
        for _, row in priority.iterrows():
            print(f"    - {row['uhf_name']}: burden={row['health_burden']:.1f}, coverage={row['coverage_pct']:.1f}%")
    
    # Highest burden neighborhood
    highest = rankings_df.iloc[0]
    print(f"  • Highest health burden: {highest['uhf_name']} (score: {highest['composite_burden_score']:.1f})")
    
    # Coverage targeting
    positive_targeting = condition_df[condition_df['coverage_gap_pp'] > 0]
    if len(positive_targeting) > 0:
        print(f"  • MSHP shows positive targeting for {len(positive_targeting)}/{len(condition_df)} conditions")
    
    print(f"\nOutputs saved to: {TABLES_DIR}")
    print("  - health_burden_composite_scores.csv")
    print("  - health_coverage_mismatch.csv")
    print("  - neighborhood_health_rankings.csv")
    print("  - condition_specific_summary.csv")


if __name__ == "__main__":
    main()

