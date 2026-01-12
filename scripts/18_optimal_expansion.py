#!/usr/bin/env python3
"""
Script 18: Optimal Expansion Planning Model

Input:
    - data/processed/bronx_schools_full.csv

Output:
    - outputs/tables/optimal_expansion_5_schools.csv
    - outputs/tables/optimal_expansion_10_schools.csv
    - outputs/tables/optimal_expansion_20_schools.csv
    - outputs/tables/expansion_scenarios_comparison.csv
    - outputs/figures/expansion_scenarios_comparison.png

This script implements facility location optimization for MSHP expansion:
1. Greedy weighted selection (simple approach)
2. Coverage maximization (geographic distribution)
3. Equity-constrained optimization
4. Scenario comparison and visualization
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mshp_gap.paths import PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, CONFIGS_DIR
from mshp_gap.io_utils import atomic_write_csv, write_metadata_sidecar
from mshp_gap.logging_utils import (
    get_logger,
    get_run_id,
    log_step_start,
    log_step_end,
    log_output_written,
)

logger = get_logger("18_optimal_expansion")


# Default expansion weights
DEFAULT_WEIGHTS = {
    'priority_score': 0.35,      # From existing priority ranking
    'enrollment': 0.20,          # More students = more impact
    'isolation_score': 0.15,     # Far from existing MSHP schools
    'health_burden': 0.20,       # Neighborhood health needs
    'equity_score': 0.10,        # Prioritize high-poverty, high-risk
}


def load_data():
    """Load the full schools dataset."""
    log_step_start(logger, "load_data")
    
    data_file = PROCESSED_DIR / "bronx_schools_full.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} schools")

    # Standardize coordinate column names for downstream functions
    if 'lat' not in df.columns and 'latitude' in df.columns:
        df['lat'] = df['latitude']
    if 'lon' not in df.columns and 'longitude' in df.columns:
        df['lon'] = df['longitude']
    
    # Separate MSHP and non-MSHP schools
    mshp = df[df['has_mshp'] == True]
    non_mshp = df[df['has_mshp'] == False]
    
    logger.info(f"MSHP schools: {len(mshp)}, Non-MSHP: {len(non_mshp)}")
    
    log_step_end(logger, "load_data")
    return df, mshp, non_mshp


def calculate_distance_matrix(df):
    """Calculate pairwise distances between all schools."""
    log_step_start(logger, "calculate_distance_matrix")
    
    coords = df[['lat', 'lon']].values
    
    # Calculate haversine-like distance (approximate miles)
    # For small areas like Bronx, Euclidean on lat/lon is acceptable
    # 1 degree lat ≈ 69 miles, 1 degree lon ≈ 53 miles at NYC latitude
    lat_miles = 69.0
    lon_miles = 53.0
    
    coords_miles = coords.copy()
    coords_miles[:, 0] *= lat_miles
    coords_miles[:, 1] *= lon_miles
    
    dist_matrix = cdist(coords_miles, coords_miles, metric='euclidean')
    
    logger.info(f"Distance matrix shape: {dist_matrix.shape}")
    logger.info(f"Max distance: {dist_matrix.max():.2f} miles")
    
    log_step_end(logger, "calculate_distance_matrix")
    return dist_matrix


def calculate_isolation_scores(non_mshp, mshp):
    """
    Calculate isolation score for each non-MSHP school.
    
    Isolation = distance to nearest existing MSHP school.
    Higher isolation = higher priority for new MSHP.
    """
    log_step_start(logger, "calculate_isolation_scores")
    
    if len(mshp) == 0:
        return pd.Series(50, index=non_mshp.index)
    
    # Get coordinates
    non_mshp_coords = non_mshp[['lat', 'lon']].values
    mshp_coords = mshp[['lat', 'lon']].values
    
    # Scale to approximate miles
    lat_miles = 69.0
    lon_miles = 53.0
    
    non_mshp_miles = non_mshp_coords.copy()
    non_mshp_miles[:, 0] *= lat_miles
    non_mshp_miles[:, 1] *= lon_miles
    
    mshp_miles = mshp_coords.copy()
    mshp_miles[:, 0] *= lat_miles
    mshp_miles[:, 1] *= lon_miles
    
    # Calculate distance to nearest MSHP
    distances = cdist(non_mshp_miles, mshp_miles, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    # Convert to percentile scores (0-100)
    isolation_scores = pd.Series(min_distances, index=non_mshp.index)
    isolation_scores = isolation_scores.rank(pct=True) * 100
    
    logger.info(f"Isolation scores calculated for {len(isolation_scores)} schools")
    logger.info(f"Max distance to nearest MSHP: {min_distances.max():.2f} miles")
    
    log_step_end(logger, "calculate_isolation_scores")
    return isolation_scores, min_distances


def calculate_composite_expansion_score(non_mshp, isolation_scores, weights=None):
    """
    Calculate weighted composite score for expansion priority.
    """
    log_step_start(logger, "calculate_composite_expansion_score")
    
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    df = non_mshp.copy()
    
    # 1. Priority score (from existing analysis - normalize)
    if 'health_burden_composite' in df.columns:
        df['priority_normalized'] = df['health_burden_composite'].rank(pct=True) * 100
    else:
        # Fallback to asthma + absenteeism
        df['priority_normalized'] = (
            df['asthma_ed_rate'].rank(pct=True) * 50 +
            df['chronic_absenteeism_rate'].rank(pct=True, na_option='bottom') * 50
        )
    
    # 2. Enrollment score (larger = higher priority)
    df['enrollment_normalized'] = df['enrollment'].rank(pct=True, na_option='bottom') * 100
    
    # 3. Isolation score (already 0-100)
    df['isolation_normalized'] = isolation_scores
    
    # 4. Health burden (already 0-100 in health_burden_composite)
    if 'health_burden_composite' in df.columns:
        df['health_normalized'] = df['health_burden_composite']
    else:
        df['health_normalized'] = df['asthma_ed_rate'].rank(pct=True) * 100
    
    # 5. Equity score (poverty + SWD rate)
    df['equity_normalized'] = (
        df['pct_poverty'].rank(pct=True) * 50 +
        df['pct_swd'].rank(pct=True, na_option='bottom') * 50
    )
    
    # Calculate weighted composite
    total_weight = sum(weights.values())
    df['expansion_score'] = (
        weights.get('priority_score', 0.35) / total_weight * df['priority_normalized'] +
        weights.get('enrollment', 0.20) / total_weight * df['enrollment_normalized'] +
        weights.get('isolation_score', 0.15) / total_weight * df['isolation_normalized'] +
        weights.get('health_burden', 0.20) / total_weight * df['health_normalized'] +
        weights.get('equity_score', 0.10) / total_weight * df['equity_normalized']
    )
    
    df['expansion_score'] = df['expansion_score'].round(2)
    
    log_step_end(logger, "calculate_composite_expansion_score")
    return df


def select_greedy_expansion(df_scored, n_schools, mshp_coords):
    """
    Greedy selection: Pick top N schools by composite score.
    """
    log_step_start(logger, "select_greedy_expansion", n_schools=n_schools)
    
    selected = df_scored.nlargest(n_schools, 'expansion_score').copy()
    selected['selection_method'] = 'greedy'
    selected['selection_rank'] = range(1, len(selected) + 1)
    
    logger.info(f"Selected {len(selected)} schools via greedy method")
    
    log_step_end(logger, "select_greedy_expansion")
    return selected


def select_coverage_optimized(df_scored, n_schools, mshp, min_distances):
    """
    Coverage-optimized selection: Balance score with geographic coverage.
    
    Iteratively select schools, updating distances after each selection.
    """
    log_step_start(logger, "select_coverage_optimized", n_schools=n_schools)
    
    df = df_scored.copy()
    df['current_min_distance'] = min_distances
    
    selected_indices = []
    selected_dbns = []
    
    # Coordinates for distance calculation
    lat_miles = 69.0
    lon_miles = 53.0
    
    mshp_coords_list = mshp[['lat', 'lon']].values.tolist() if len(mshp) > 0 else []
    
    for i in range(n_schools):
        # Calculate combined score: expansion_score + isolation bonus
        # Schools far from coverage get a bonus
        df['coverage_adjusted_score'] = (
            df['expansion_score'] * 0.7 +
            df['current_min_distance'].rank(pct=True) * 100 * 0.3
        )
        
        # Select best remaining school
        remaining = df[~df['dbn'].isin(selected_dbns)]
        if len(remaining) == 0:
            break
            
        best_idx = remaining['coverage_adjusted_score'].idxmax()
        best_school = df.loc[best_idx]
        
        selected_indices.append(best_idx)
        selected_dbns.append(best_school['dbn'])
        
        # Update distances - add this school to "covered" list
        new_mshp_coords = np.array([[
            best_school['lat'] * lat_miles,
            best_school['lon'] * lon_miles
        ]])
        
        # Recalculate minimum distances
        all_coords = df[['lat', 'lon']].values
        all_coords_miles = all_coords.copy()
        all_coords_miles[:, 0] *= lat_miles
        all_coords_miles[:, 1] *= lon_miles
        
        new_distances = cdist(all_coords_miles, new_mshp_coords, metric='euclidean').flatten()
        df['current_min_distance'] = np.minimum(df['current_min_distance'], new_distances)
    
    selected = df.loc[selected_indices].copy()
    selected['selection_method'] = 'coverage_optimized'
    selected['selection_rank'] = range(1, len(selected) + 1)
    
    logger.info(f"Selected {len(selected)} schools via coverage-optimized method")
    
    log_step_end(logger, "select_coverage_optimized")
    return selected


def select_equity_constrained(df_scored, n_schools, mshp):
    """
    Equity-constrained selection: Ensure coverage across neighborhoods.
    
    Guarantees at least one new MSHP per neighborhood (if n_schools >= # neighborhoods).
    """
    log_step_start(logger, "select_equity_constrained", n_schools=n_schools)
    
    df = df_scored.copy()
    
    # Get neighborhoods
    neighborhoods = df['uhf_name'].unique()
    n_neighborhoods = len(neighborhoods)
    
    selected_dbns = []
    
    # Phase 1: Select best school from each neighborhood
    per_neighborhood = n_schools // n_neighborhoods if n_schools >= n_neighborhoods else 0
    
    if per_neighborhood > 0:
        for neighborhood in neighborhoods:
            neigh_schools = df[(df['uhf_name'] == neighborhood) & (~df['dbn'].isin(selected_dbns))]
            if len(neigh_schools) > 0:
                # Select top school in this neighborhood
                best = neigh_schools.nlargest(1, 'expansion_score')
                selected_dbns.append(best['dbn'].values[0])
    
    # Phase 2: Fill remaining slots with highest scoring schools
    remaining_slots = n_schools - len(selected_dbns)
    if remaining_slots > 0:
        remaining_schools = df[~df['dbn'].isin(selected_dbns)]
        additional = remaining_schools.nlargest(remaining_slots, 'expansion_score')
        selected_dbns.extend(additional['dbn'].tolist())
    
    selected = df[df['dbn'].isin(selected_dbns)].copy()
    selected = selected.sort_values('expansion_score', ascending=False)
    selected['selection_method'] = 'equity_constrained'
    selected['selection_rank'] = range(1, len(selected) + 1)
    
    logger.info(f"Selected {len(selected)} schools via equity-constrained method")
    
    # Log neighborhood distribution
    neigh_counts = selected['uhf_name'].value_counts()
    logger.info(f"Neighborhood distribution: {neigh_counts.to_dict()}")
    
    log_step_end(logger, "select_equity_constrained")
    return selected


def evaluate_expansion_scenario(selected, all_schools, mshp):
    """
    Evaluate the impact of an expansion scenario.
    """
    # Calculate metrics
    n_selected = len(selected)
    total_new_students = selected['enrollment'].sum()
    avg_health_burden = selected['health_burden_composite'].mean() if 'health_burden_composite' in selected.columns else None
    neighborhoods_covered = selected['uhf_name'].nunique()
    
    # New coverage rate
    new_total_mshp = len(mshp) + n_selected
    new_coverage_rate = new_total_mshp / len(all_schools) * 100
    
    # Average distance reduction (simplified)
    # Before: avg distance to nearest MSHP
    # After: would decrease as we add more MSHP schools
    
    return {
        'schools_added': n_selected,
        'students_reached': int(total_new_students),
        'avg_health_burden': round(avg_health_burden, 1) if avg_health_burden else None,
        'neighborhoods_covered': neighborhoods_covered,
        'new_coverage_rate_pct': round(new_coverage_rate, 1),
        'old_coverage_rate_pct': round(len(mshp) / len(all_schools) * 100, 1),
    }


def create_expansion_output(selected, method_name, n_schools):
    """
    Create output DataFrame for expansion scenario.
    """
    output_cols = [
        'selection_rank', 'dbn', 'school_name', 'school_type', 'address',
        'uhf_name', 'enrollment', 'chronic_absenteeism_rate',
        'asthma_ed_rate', 'pct_poverty', 'expansion_score', 'selection_method'
    ]
    
    available_cols = [c for c in output_cols if c in selected.columns]
    output = selected[available_cols].copy()
    
    # Format percentages
    if 'pct_poverty' in output.columns:
        output['pct_poverty'] = (output['pct_poverty'] * 100).round(1)
    if 'chronic_absenteeism_rate' in output.columns:
        output['chronic_absenteeism_rate'] = output['chronic_absenteeism_rate'].round(1)
    
    return output


def main():
    """Main entry point for optimal expansion planning."""
    logger.info("=" * 60)
    logger.info("OPTIMAL EXPANSION PLANNING MODEL")
    logger.info("=" * 60)
    
    run_id = get_run_id()
    
    # Load data
    df, mshp, non_mshp = load_data()
    
    # Calculate isolation scores
    logger.info("\n--- Calculating Isolation Scores ---")
    isolation_scores, min_distances = calculate_isolation_scores(non_mshp, mshp)
    
    # Calculate composite expansion scores
    logger.info("\n--- Calculating Expansion Scores ---")
    df_scored = calculate_composite_expansion_score(non_mshp, isolation_scores)
    
    # Expansion scenarios
    scenarios = [5, 10, 20]
    methods = ['greedy', 'coverage_optimized', 'equity_constrained']
    
    all_results = []
    
    print("\n" + "=" * 70)
    print("OPTIMAL EXPANSION SCENARIOS")
    print("=" * 70)
    
    for n_schools in scenarios:
        logger.info(f"\n--- Scenario: Add {n_schools} Schools ---")
        print(f"\n{'='*50}")
        print(f"SCENARIO: ADD {n_schools} MSHP SCHOOLS")
        print(f"{'='*50}")
        
        # Method 1: Greedy
        selected_greedy = select_greedy_expansion(df_scored, n_schools, mshp[['lat', 'lon']].values)
        eval_greedy = evaluate_expansion_scenario(selected_greedy, df, mshp)
        eval_greedy['method'] = 'greedy'
        eval_greedy['n_schools'] = n_schools
        all_results.append(eval_greedy)
        
        # Method 2: Coverage-optimized
        selected_coverage = select_coverage_optimized(df_scored, n_schools, mshp, min_distances)
        eval_coverage = evaluate_expansion_scenario(selected_coverage, df, mshp)
        eval_coverage['method'] = 'coverage_optimized'
        eval_coverage['n_schools'] = n_schools
        all_results.append(eval_coverage)
        
        # Method 3: Equity-constrained
        selected_equity = select_equity_constrained(df_scored, n_schools, mshp)
        eval_equity = evaluate_expansion_scenario(selected_equity, df, mshp)
        eval_equity['method'] = 'equity_constrained'
        eval_equity['n_schools'] = n_schools
        all_results.append(eval_equity)
        
        # Write a single output file containing ALL methods for this N
        greedy_output = create_expansion_output(selected_greedy, 'greedy', n_schools)
        greedy_output['method'] = 'greedy'
        coverage_output = create_expansion_output(selected_coverage, 'coverage_optimized', n_schools)
        coverage_output['method'] = 'coverage_optimized'
        equity_output = create_expansion_output(selected_equity, 'equity_constrained', n_schools)
        equity_output['method'] = 'equity_constrained'

        output = pd.concat([greedy_output, coverage_output, equity_output], ignore_index=True)

        output_path = TABLES_DIR / f"optimal_expansion_{n_schools}_schools.csv"
        atomic_write_csv(output_path, output)
        write_metadata_sidecar(
            data_path=output_path,
            script_name="18_optimal_expansion.py",
            run_id=run_id,
            description=f"Top {n_schools} schools recommended for MSHP expansion (all methods)",
            inputs=["bronx_schools_full.csv"],
            row_count=len(output),
            columns=list(output.columns),
            weights=DEFAULT_WEIGHTS,
            evaluation={
                "greedy": eval_greedy,
                "coverage_optimized": eval_coverage,
                "equity_constrained": eval_equity,
            },
        )
        log_output_written(logger, output_path, row_count=len(output))
        
        # Print summary
        print(f"\nMethod Comparison for {n_schools} Schools:")
        print(f"  {'Method':<20} {'Students':<12} {'Neighborhoods':<15} {'New Coverage':<12}")
        print(f"  {'-'*60}")
        print(f"  {'Greedy':<20} {eval_greedy['students_reached']:>10,} {eval_greedy['neighborhoods_covered']:>12} {eval_greedy['new_coverage_rate_pct']:>10.1f}%")
        print(f"  {'Coverage-Optimized':<20} {eval_coverage['students_reached']:>10,} {eval_coverage['neighborhoods_covered']:>12} {eval_coverage['new_coverage_rate_pct']:>10.1f}%")
        print(f"  {'Equity-Constrained':<20} {eval_equity['students_reached']:>10,} {eval_equity['neighborhoods_covered']:>12} {eval_equity['new_coverage_rate_pct']:>10.1f}%")
        
        print(f"\nTop 10 Recommended Schools (Greedy Method):")
        display_cols = ['selection_rank', 'school_name', 'uhf_name', 'enrollment', 'expansion_score']
        available = [c for c in display_cols if c in greedy_output.columns]
        print(greedy_output.head(10)[available].to_string(index=False))
    
    # Save comparison summary
    comparison_df = pd.DataFrame(all_results)
    
    output_path = TABLES_DIR / "expansion_scenarios_comparison.csv"
    atomic_write_csv(output_path, comparison_df)
    write_metadata_sidecar(
        data_path=output_path,
        script_name="18_optimal_expansion.py",
        run_id=run_id,
        description="Comparison of expansion scenarios and methods",
        inputs=["bronx_schools_full.csv"],
        row_count=len(comparison_df),
        columns=list(comparison_df.columns),
    )
    log_output_written(logger, output_path, row_count=len(comparison_df))
    
    # Create visualization
    create_expansion_visualization(comparison_df, df_scored, mshp)
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPANSION PLANNING COMPLETE")
    print("=" * 70)
    
    print("\n📊 KEY RECOMMENDATIONS:")
    
    # Highlight consensus schools (appear in all scenarios)
    top_5 = df_scored.nlargest(5, 'expansion_score')
    print(f"\n  Top 5 Priority Schools (Consensus):")
    for i, (_, school) in enumerate(top_5.iterrows(), 1):
        print(f"    {i}. {school['school_name'][:40]}")
        print(f"       Neighborhood: {school['uhf_name']}, Enrollment: {school['enrollment']:.0f}")
    
    print(f"\n  Current MSHP Coverage: {len(mshp)}/{len(df)} schools ({len(mshp)/len(df)*100:.1f}%)")
    print(f"  With +20 schools: {len(mshp)+20}/{len(df)} schools ({(len(mshp)+20)/len(df)*100:.1f}%)")
    
    print(f"\nOutputs saved to: {TABLES_DIR}")
    for n in scenarios:
        print(f"  - optimal_expansion_{n}_schools.csv")
    print(f"  - expansion_scenarios_comparison.csv")


def create_expansion_visualization(comparison_df, df_scored, mshp):
    """Create visualization comparing expansion scenarios."""
    import matplotlib.pyplot as plt
    
    print("\nCreating expansion visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('MSHP Expansion Scenario Analysis', fontsize=16, fontweight='bold')
    
    colors = {'greedy': '#1565C0', 'coverage_optimized': '#2E7D32', 'equity_constrained': '#FF8F00'}
    
    # 1. Students reached by scenario
    ax1 = axes[0, 0]
    
    for method in ['greedy', 'coverage_optimized', 'equity_constrained']:
        method_data = comparison_df[comparison_df['method'] == method]
        ax1.plot(method_data['n_schools'], method_data['students_reached'], 
                marker='o', linewidth=2, markersize=8, label=method.replace('_', ' ').title(),
                color=colors[method])
    
    ax1.set_xlabel('Schools Added')
    ax1.set_ylabel('Students Reached')
    ax1.set_title('Additional Students with MSHP Access')
    ax1.legend()
    ax1.set_xticks([5, 10, 20])
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # 2. Coverage rate improvement
    ax2 = axes[0, 1]
    
    current_coverage = comparison_df['old_coverage_rate_pct'].iloc[0]
    
    for method in ['greedy', 'coverage_optimized', 'equity_constrained']:
        method_data = comparison_df[comparison_df['method'] == method]
        ax2.plot(method_data['n_schools'], method_data['new_coverage_rate_pct'], 
                marker='s', linewidth=2, markersize=8, label=method.replace('_', ' ').title(),
                color=colors[method])
    
    ax2.axhline(current_coverage, linestyle='--', color='gray', label=f'Current ({current_coverage:.1f}%)')
    ax2.set_xlabel('Schools Added')
    ax2.set_ylabel('Coverage Rate (%)')
    ax2.set_title('School Coverage Rate by Scenario')
    ax2.legend()
    ax2.set_xticks([5, 10, 20])
    
    # 3. Top 20 schools by expansion score
    ax3 = axes[1, 0]
    
    top_20 = df_scored.nlargest(20, 'expansion_score')
    y = range(len(top_20))
    
    colors_bar = ['#1565C0' if i < 5 else '#42A5F5' if i < 10 else '#90CAF9' for i in range(20)]
    
    ax3.barh(y, top_20['expansion_score'], color=colors_bar, alpha=0.8)
    ax3.set_yticks(y)
    ax3.set_yticklabels([n[:25] for n in top_20['school_name']], fontsize=8)
    ax3.set_xlabel('Expansion Priority Score')
    ax3.set_title('Top 20 Schools for MSHP Expansion')
    ax3.invert_yaxis()
    
    # Add tier markers
    ax3.axvline(top_20.iloc[4]['expansion_score'], linestyle='--', color='green', alpha=0.5)
    ax3.axvline(top_20.iloc[9]['expansion_score'], linestyle='--', color='orange', alpha=0.5)
    
    # 4. Neighborhood distribution
    ax4 = axes[1, 1]
    
    top_20_by_neigh = top_20['uhf_name'].value_counts()
    
    ax4.pie(top_20_by_neigh.values, labels=[n.split(' - ')[0] for n in top_20_by_neigh.index],
            autopct='%1.0f%%', colors=plt.cm.Set3.colors[:len(top_20_by_neigh)])
    ax4.set_title('Top 20 Schools by Neighborhood')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'expansion_scenarios_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {FIGURES_DIR / 'expansion_scenarios_comparison.png'}")


if __name__ == "__main__":
    main()

