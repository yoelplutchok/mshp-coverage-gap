# Data Dictionary

## MSHP Coverage Gap Analysis - Bronx Schools

---

## Primary Analysis Dataset

### `data/processed/bronx_schools_full.csv`

The comprehensive analysis-ready dataset containing all school, demographic, health, and analysis variables.

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `dbn` | string | District-Borough-Number unique school identifier | NYC DOE | "07X259" |
| `school_name` | string | Official school name | NYC DOE | "P.S. 048 Joseph R. Drake" |
| `latitude` | float | School latitude coordinate (WGS84) | NYC DOE | 40.8175 |
| `longitude` | float | School longitude coordinate (WGS84) | NYC DOE | -73.8892 |
| `address` | string | School street address | NYC DOE | "1290 Spofford Avenue" |
| `school_type` | string | School level classification | Derived | "Elementary", "Middle", "High", "K-8" |
| `enrollment` | int | Total student enrollment | NYC DOE | 485 |
| `grades` | string | Grade span served | NYC DOE | "PK-5" |

#### MSHP Coverage

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `has_mshp` | bool | Whether school has MSHP clinic | Montefiore/DOE SBHC | True/False |
| `clinic_name` | string | Name of MSHP clinic (if covered) | Montefiore | "Morris Educational Campus" |
| `expansion_cohort` | string | MSHP expansion wave (for causal analysis) | Estimated | "cohort_2_2010_2015" |
| `estimated_start_year` | int | Estimated year MSHP coverage began | Estimated | 2012 |
| `years_with_mshp` | int | Years of MSHP coverage | Calculated | 12 |

#### Geographic Assignment

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `uhf_code` | int | United Hospital Fund neighborhood code | Spatial join | 107 |
| `uhf_name` | string | UHF neighborhood name | NYC DOHMH | "Hunts Point - Mott Haven" |

#### Attendance Data

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `chronic_absenteeism_rate` | float | % of students missing 10%+ of school days | NYC DOE InfoHub | 42.3 |

#### School Demographics

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `pct_poverty` | float | Proportion of students in poverty (0-1) | NYC Open Data | 0.92 |
| `pct_ell` | float | Proportion English Language Learners | NYC Open Data | 0.28 |
| `pct_swd` | float | Proportion Students with Disabilities | NYC Open Data | 0.22 |
| `pct_black` | float | Proportion Black students | NYC Open Data | 0.30 |
| `pct_hispanic` | float | Proportion Hispanic students | NYC Open Data | 0.65 |
| `pct_white` | float | Proportion White students | NYC Open Data | 0.02 |
| `pct_asian` | float | Proportion Asian students | NYC Open Data | 0.01 |

#### Health Outcomes (Neighborhood-Level)

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `asthma_ed_rate` | float | Asthma ED visits per 10,000 children (5-17) | NYC DOHMH EHDP | 192.3 |
| `asthma_hosp_rate` | float | Asthma hospitalizations per 10,000 | NYC DOHMH | 45.2 |
| `childhood_obesity_pct` | float | % children overweight/obese (K-8) | NYC Fitnessgram | 34.5 |
| `youth_mental_health_ed_rate` | float | Youth mental health ED visits per 10,000 | NYC DOHMH | 168.4 |
| `lead_elevated_pct` | float | % children with elevated blood lead | NYC DOHMH | 4.5 |
| `teen_pregnancy_rate` | float | Teen births per 1,000 females 15-19 | NYC DOHMH | 48.2 |
| `preventable_hosp_rate` | float | Preventable hospitalizations per 100,000 | NYC DOHMH | 268.5 |
| `food_insecurity_pct` | float | % households food insecure | Community Health | 38.2 |

#### Social Vulnerability Index

| Column | Type | Description | Source | Example |
|--------|------|-------------|--------|---------|
| `svi_socioeconomic` | float | SVI Theme 1: Socioeconomic (0-1) | CDC SVI/Synthetic | 0.85 |
| `svi_household_disability` | float | SVI Theme 2: Household/Disability | CDC SVI/Synthetic | 0.72 |
| `svi_minority_language` | float | SVI Theme 3: Minority/Language | CDC SVI/Synthetic | 0.91 |
| `svi_housing_transport` | float | SVI Theme 4: Housing/Transport | CDC SVI/Synthetic | 0.68 |
| `svi_overall` | float | Overall SVI score (0-1, higher = more vulnerable) | CDC SVI/Synthetic | 0.79 |

#### Computed Scores

| Column | Type | Description | Formula | Example |
|--------|------|-------------|---------|---------|
| `priority_score` | float | Expansion priority score (0-100) | See methodology | 88.5 |
| `priority_tier` | string | Priority classification | Based on percentile | "Tier 1 - Critical" |
| `health_burden_composite` | float | Composite health burden (0-100) | Weighted average of health indicators | 85.3 |
| `distance_to_nearest_mshp` | float | Distance to nearest MSHP school (km) | Haversine | 1.24 |

---

## Priority Ranking Output

### `outputs/tables/non_mshp_schools_priority_ranked.csv`

Ranked list of non-MSHP schools for program expansion.

| Column | Type | Description |
|--------|------|-------------|
| `rank` | int | Priority rank (1 = highest) |
| `dbn` | string | School identifier |
| `school_name` | string | School name |
| `uhf_name` | string | Neighborhood |
| `priority_score` | float | Composite priority score |
| `priority_tier` | string | Tier classification |
| `asthma_ed_rate` | float | Neighborhood asthma rate |
| `chronic_absenteeism_rate` | float | School absenteeism rate |
| `enrollment` | int | School enrollment |

---

## Neighborhood Summary

### `outputs/tables/neighborhood_summary.csv`

Aggregated statistics by UHF neighborhood.

| Column | Type | Description |
|--------|------|-------------|
| `uhf_name` | string | Neighborhood name |
| `total_schools` | int | Total schools in neighborhood |
| `mshp_schools` | int | Schools with MSHP coverage |
| `mshp_coverage_pct` | float | Percentage with coverage |
| `asthma_ed_rate` | float | Asthma ED visit rate |
| `mean_absenteeism` | float | Mean chronic absenteeism |
| `gap_score` | float | Coverage gap score |

---

## Causal Inference Outputs

### `outputs/tables/propensity_scores.csv`

Propensity scores for causal analysis.

| Column | Type | Description |
|--------|------|-------------|
| `dbn` | string | School identifier |
| `school_name` | string | School name |
| `has_mshp` | bool | MSHP coverage status |
| `propensity_score` | float | Probability of having MSHP (0-1) |
| `ps_quintile` | string | Propensity score quintile (Q1-Q5) |
| `chronic_absenteeism_rate` | float | Absenteeism rate |

### `outputs/tables/causal_inference_results.csv`

Summary of causal effect estimates.

| Column | Type | Description |
|--------|------|-------------|
| `method` | string | Estimation method |
| `mshp_effect` | float | Estimated MSHP effect (% points) |
| `interpretation` | string | Plain-language interpretation |

---

## Data Quality Notes

### Missing Values
- Schools missing attendance data: ~10% (new schools, special programs)
- Schools missing demographics: 5% (filled with neighborhood estimates)
- All schools have geographic assignments

### Known Issues
- 12 duplicate DBNs in original DOE source data (removed during processing)
- 7 MSHP schools not in DOE file (5 charters + 2 special programs)
- SVI data is synthetic (CDC download failed; based on neighborhood asthma rates)
- MSHP expansion history is estimated, not verified

### Temporal Coverage
- School locations: 2024-25 academic year
- Attendance data: 2024-25 academic year
- Asthma data: 2022 (most recent available)
- Demographics: 2021-22 (most recent in API)

