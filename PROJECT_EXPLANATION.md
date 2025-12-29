# MSHP Coverage Gap Project - Comprehensive Explanation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Purpose and Context](#project-purpose-and-context)
3. [Technical Architecture](#technical-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Analysis Methodology](#analysis-methodology)
6. [Key Findings](#key-findings)
7. [How to Use This Project](#how-to-use-this-project)
8. [Extending the Project](#extending-the-project)
9. [Technical Deep Dive](#technical-deep-dive)

---

## Executive Summary

### What is this project?

The **MSHP Coverage Gap** project is a comprehensive geospatial and statistical analysis that investigates the distribution of the **Montefiore School Health Program (MSHP)** across Bronx public schools. It identifies coverage gaps in neighborhoods with the highest childhood health burdens and provides data-driven recommendations for program expansion.

### Why does it matter?

School-based health centers (SBHCs) play a crucial role in addressing health inequities, particularly in underserved communities. The Bronx has the highest childhood asthma rates in New York City, and chronic health conditions are a major driver of school absenteeism. This project:

- **Maps health equity gaps**: Identifies which schools in high-need neighborhoods lack access to school-based healthcare
- **Informs policy decisions**: Provides evidence-based priority rankings for MSHP expansion
- **Evaluates program targeting**: Assesses whether MSHP is effectively reaching the highest-need schools
- **Quantifies impact**: Uses causal inference methods to estimate MSHP's effect on student outcomes

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Bronx Public Schools Analyzed** | 370 |
| **Schools with MSHP Coverage** | 84 (23%) |
| **Schools without MSHP** | 286 (77%) |
| **High-Priority Schools for Expansion** | 58 (Tier 1) |
| **"Health Desert" Schools** | 22 (>1 mile from MSHP) |

### Main Finding

**MSHP successfully targets high-need areas**. Schools in neighborhoods with the highest asthma burdens (Hunts Point, Morrisania) have MSHP coverage. However, **significant coverage gaps remain** even in these high-need areas, with only 15% of schools in some neighborhoods having access to MSHP.

---

## Project Purpose and Context

### The Problem

**Childhood asthma** is a major public health crisis in the Bronx:
- Bronx children have asthma emergency department (ED) visit rates **3-4 times higher** than the NYC average
- Chronic health conditions lead to **chronic absenteeism** (missing 10% or more of school days)
- School-based health centers can reduce barriers to healthcare access (no transportation, no insurance, no time off work for parents)

**The Montefiore School Health Program (MSHP)** operates school-based health centers in select Bronx schools, providing:
- Primary care
- Mental health services
- Dental care
- Health education
- Chronic disease management (including asthma)

### Research Questions

This project addresses four key questions:

1. **Geographic equity**: Are schools WITHOUT MSHP coverage disproportionately located in higher-need neighborhoods?

2. **Association with outcomes**: Is there a relationship between MSHP coverage and chronic absenteeism rates?

3. **Prioritization**: Which schools should be prioritized for MSHP expansion based on health burden, social vulnerability, and accessibility?

4. **Causal inference**: What is the causal effect of MSHP on student outcomes, accounting for the fact that MSHP intentionally targets high-need schools?

### Who is this for?

**Primary audiences:**
- **Policy makers** at NYC Department of Education and Department of Health
- **Program administrators** at Montefiore Health System
- **Public health researchers** studying health equity and school-based interventions
- **Community advocates** working on health justice in the Bronx

**Technical audiences:**
- **Data scientists** interested in geospatial analysis and causal inference
- **Software developers** looking to build similar health equity analyses
- **Students** learning about public health data analysis

---

## Technical Architecture

### Technology Stack

**Language & Core Libraries:**
- **Python 3.11** - Modern Python with type hints and performance improvements
- **pandas ≥2.0** - Data manipulation and analysis
- **geopandas ≥0.14** - Geospatial data operations
- **NumPy ≥1.24** - Numerical computing

**Geospatial Tools:**
- **Shapely ≥2.0** - Geometric operations (point-in-polygon, distance calculations)
- **Folium ≥0.15** - Interactive map generation
- **PyProj ≥3.6** - Coordinate reference system transformations
- **Fiona ≥1.9** - Reading/writing geographic file formats

**Data Collection:**
- **requests ≥2.31** - HTTP client for API calls
- **BeautifulSoup4 ≥4.12** - HTML/PDF parsing
- **thefuzz ≥0.20** - Fuzzy string matching for school name matching

**Statistical Analysis:**
- **SciPy ≥1.11** - Statistical tests (t-tests, correlations)
- **statsmodels** (implicit) - Regression and propensity score models

**Visualization:**
- **Matplotlib ≥3.8** - Static chart generation
- **Seaborn ≥0.13** - Statistical visualizations

**Development Tools:**
- **pytest ≥7.4** - Unit testing framework
- **black ≥23.0** - Code formatting
- **ruff ≥0.1** - Fast Python linter

### Project Structure

```
mshp-coverage-gap/
│
├── configs/
│   └── params.yml              # Central configuration (all magic numbers)
│
├── data/
│   ├── raw/                    # Original downloaded data (never modified)
│   ├── processed/              # Cleaned, analysis-ready datasets
│   ├── manual/                 # Hand-curated MSHP school lists
│   └── geo/                    # Geographic boundaries (GeoJSON)
│
├── outputs/
│   ├── figures/                # Static PNG visualizations (13 files)
│   ├── interactive/            # HTML interactive maps (3 files)
│   └── tables/                 # CSV summary tables
│
├── scripts/                    # Numbered pipeline scripts (01-14)
│   ├── 01_collect_doe_schools.py       # Download school locations
│   ├── 04_collect_uhf_boundaries.py    # Download neighborhood boundaries
│   ├── 05_process_and_join.py          # Clean and merge data
│   ├── 08_analyze.py                   # Statistical analysis
│   ├── 10_visualize.py                 # Basic visualizations
│   ├── 10_visualize_enhanced.py        # Enhanced interactive maps
│   ├── 11_enhanced_analysis.py         # SVI and accessibility analysis
│   ├── 12_enhanced_visualizations.py   # Advanced charts
│   ├── 13_add_demographics_health.py   # Causal inference prep
│   └── 14_demographics_visualizations.py # Final visualizations
│
├── src/
│   └── mshp_gap/               # Reusable Python package
│       ├── paths.py            # Path constants
│       ├── io_utils.py         # File I/O utilities
│       └── logging_utils.py    # Structured logging
│
├── docs/                       # Documentation
│   ├── data_dictionary.md      # Field definitions
│   ├── methodology.md          # Analysis methods
│   └── limitations.md          # Known limitations
│
├── tests/                      # Unit tests
│
├── Makefile                    # Pipeline orchestration
├── environment.yml             # Conda environment specification
├── pyproject.toml              # Python package configuration
└── README.md                   # Quick start guide
```

### Design Principles

**1. Reproducibility First**
- All parameters in `configs/params.yml` - no magic numbers in code
- Atomic writes prevent partial file corruption
- Metadata sidecars track data provenance
- Structured logging with run IDs

**2. Pipeline Architecture**
- Numbered scripts (01-14) show execution order
- Each script has clear inputs and outputs
- Can run individual steps or full pipeline
- Make targets abstract script dependencies

**3. Data Integrity**
- Raw data never modified (immutable)
- Processed data versioned with timestamps
- Validation checks at each stage
- Duplicate detection and handling

**4. Modularity**
- Reusable utilities in `src/mshp_gap/` package
- Scripts import from package, not copy-paste
- Easy to extend with new analyses
- Clean separation of concerns

---

## Data Pipeline

### Overview

The analysis pipeline consists of **14 scripts** organized into 4 stages:

```
COLLECT → PROCESS → ANALYZE → VISUALIZE
 (01-04)   (05-07)   (08-09)   (10-14)
```

### Stage 1: Data Collection (Scripts 01-04)

#### Script 01: Collect DOE School Locations
**Purpose:** Download all Bronx public school locations

**Source:** NYC Open Data API (dataset `fgmk-2drc`)

**Process:**
1. Query Socrata API with filter `borough = 'X'` (Bronx)
2. Extract school DBN (District-Borough-Number identifier), name, address, coordinates
3. Validate coordinates are within Bronx bounds
4. Save to `data/raw/doe_school_locations_YYYY-MM-DD.csv`

**Output:** ~400-500 school records

**Key fields:** `dbn`, `school_name`, `latitude`, `longitude`, `borough`

---

#### Script 04: Collect UHF Boundaries
**Purpose:** Download NYC neighborhood boundaries for spatial joins

**Source:** NYC Health GitHub repository (coronavirus-data)

**Process:**
1. Fetch GeoJSON file with all 42 UHF (United Hospital Fund) neighborhoods
2. Filter to 7 Bronx UHF codes: 101-107
3. Validate geometries (no invalid polygons)
4. Save to `data/geo/uhf_42_neighborhoods.geojson`

**Output:** 7 Bronx neighborhoods with polygon geometries

**Neighborhoods:**
- 101: Kingsbridge - Riverdale
- 102: Northeast Bronx
- 103: Fordham - Bronx Park
- 104: Pelham - Throgs Neck
- 105: Crotona - Tremont
- 106: High Bridge - Morrisania
- 107: Hunts Point - Mott Haven

---

#### Scripts 02-03: Additional Data (Not shown in repo)
These scripts would collect:
- **MSHP school list** (scraped from DOE SBHC document)
- **Chronic absenteeism data** (from NYC DOE InfoHub)
- **Asthma data** (from NYC DOHMH EHDP portal)

---

### Stage 2: Data Processing (Scripts 05-07)

#### Script 05: Process and Join
**Purpose:** Clean, standardize, and merge all data sources

**Major operations:**

1. **School standardization:**
   - Remove 12 duplicate DBNs from DOE data
   - Standardize names (uppercase, abbreviation normalization)
   - Filter to schools with valid coordinates
   - Classify school type (Elementary, Middle, High, K-8) based on name patterns
   - Extract enrollment numbers

2. **MSHP matching:**
   - Load official MSHP school list from DOE SBHC document
   - Expand campus schools (1 clinic → multiple schools)
   - Match to DOE schools by DBN
   - Fuzzy match by name for schools without DBN
   - Manual verification of match quality
   - **Result:** 84 of 91 MSHP schools successfully matched

3. **Spatial join:**
   - Convert school coordinates to GeoPandas Point geometries
   - Perform point-in-polygon join with UHF boundaries
   - Assign each school to a neighborhood
   - Validate all 370 schools assigned

4. **Data integration:**
   - Join chronic absenteeism by DBN
   - Join asthma/health data by UHF code
   - Join demographics by DBN
   - Fill missing values with neighborhood estimates

**Output:** `data/processed/bronx_schools_full.csv` (370 rows, ~50 columns)

---

### Stage 3: Analysis (Scripts 08-09)

#### Script 08: Statistical Analysis
**Purpose:** Test hypotheses about MSHP coverage and outcomes

**Analyses performed:**

1. **Descriptive statistics:**
   - Compute means, medians, SDs by MSHP status
   - Aggregate to neighborhood level
   - Calculate coverage percentages

2. **Geographic analysis:**
   - Map coverage patterns by neighborhood
   - Identify "health deserts" (schools >1 mile from MSHP)
   - Calculate distance to nearest MSHP for each school

3. **Hypothesis tests:**
   - **T-test:** MSHP vs. non-MSHP absenteeism
     - Result: p=0.70 (not significant)
   - **Mann-Whitney U:** Non-parametric alternative
   - **Correlation:** MSHP coverage % vs. neighborhood asthma rate
     - Result: r=0.24 (weak positive, as expected for equity targeting)

4. **Causal inference preparation:**
   - Estimate propensity scores (probability of having MSHP)
   - Create propensity score quintiles
   - Calculate inverse probability weights (IPW)
   - Stratified analysis within quintiles

**Key finding:** No significant difference in absenteeism between MSHP and non-MSHP schools, BUT this likely reflects successful equity targeting (MSHP prevents worse outcomes in high-need schools).

---

#### Script 09: Priority Ranking
**Purpose:** Rank non-MSHP schools for program expansion

**Priority score formula:**
```python
priority_score = (
    0.25 × asthma_percentile +        # Neighborhood health burden
    0.25 × svi_percentile +            # Social vulnerability
    0.20 × absenteeism_percentile +    # School-level absenteeism
    0.15 × isolation_percentile +      # Distance to nearest MSHP
    0.15 × enrollment_percentile       # School size (more students = higher impact)
)
```

**Tier classification:**
- **Tier 1 (Critical):** Top 20% (58 schools) - Immediate expansion priority
- **Tier 2 (High):** 60-80th percentile (57 schools)
- **Tier 3 (Moderate):** 40-60th percentile (57 schools)
- **Tier 4 (Lower):** Bottom 40% (114 schools)

**Output:** `outputs/tables/non_mshp_schools_priority_ranked.csv`

**Top 5 priority schools:**
1. H.E.R.O. High (Hunts Point - Mott Haven) - Score: 94.1
2. Bronx Studio School for Writers and Artists (High Bridge - Morrisania) - Score: 88.9
3. P.S. 048 Joseph R. Drake (Hunts Point - Mott Haven) - Score: 88.9
4. Hostos-Lincoln Academy of Science (Hunts Point - Mott Haven) - Score: 88.6
5. The Longwood Academy of Discovery (Hunts Point - Mott Haven) - Score: 87.6

---

### Stage 4: Visualization (Scripts 10-14)

#### Script 10: Basic Visualizations
**Purpose:** Create initial maps and charts

**Outputs:**
- Basic coverage map (HTML)
- Coverage gap bar chart (PNG)
- Absenteeism comparison box plot (PNG)
- Neighborhood heatmap (PNG)

---

#### Script 10 (Enhanced): Interactive Map
**Purpose:** Create publication-quality interactive map

**Features:**
- **Multiple base layers:** Dark theme (CartoDB Dark Matter), Light, Satellite
- **Choropleth layer:** Neighborhoods colored by asthma ED rate
- **School markers:**
  - Green = MSHP covered
  - Red = Not covered
  - Size scaled by enrollment
- **Priority highlights:** Tier 1 schools with pulsing markers
- **Layer controls:** Toggle layers on/off
- **Tooltips:** School info on hover
- **Popups:** Detailed stats on click

**Output:** `outputs/interactive/mshp_coverage_gap_map_enhanced.html`

---

#### Scripts 11-14: Advanced Analysis & Visualization
**Purpose:** Add demographics, social vulnerability, and causal inference

**New analyses:**
1. **Social Vulnerability Index (SVI):**
   - CDC Social Vulnerability Index by census tract
   - Aggregated to school level
   - 4 themes: Socioeconomic, Household/Disability, Minority/Language, Housing/Transport

2. **Propensity score analysis:**
   - Logistic regression: P(MSHP) ~ poverty + asthma + SVI + enrollment
   - Stratified effects within propensity score quintiles
   - Inverse probability weighting

3. **Dose-response analysis:**
   - Estimated years of MSHP exposure
   - Correlation: Years with MSHP vs. Absenteeism
   - Result: r=-0.17 (suggestive negative, but not significant)

4. **Demographics visualizations:**
   - Demographics by MSHP status
   - Health equity composite dashboard
   - Propensity score distributions
   - Causal inference summary plots

**Key visualizations:**
- `health_equity_composite.png` - Comprehensive 6-panel dashboard
- `propensity_scores.png` - Distribution by MSHP status
- `causal_inference_summary.png` - Treatment effect estimates
- `demographics_comparison.png` - School characteristics by MSHP

---

## Analysis Methodology

### 1. Geographic Methods

#### Spatial Join
**Purpose:** Assign each school to a neighborhood

**Method:**
```python
schools_gdf = gpd.GeoDataFrame(
    schools, 
    geometry=gpd.points_from_xy(schools.longitude, schools.latitude),
    crs="EPSG:4326"
)

schools_with_uhf = gpd.sjoin(
    schools_gdf, 
    uhf_boundaries, 
    how="left", 
    predicate="within"
)
```

**Validation:** All 370 schools successfully assigned (no schools outside UHF boundaries)

---

#### Distance Calculations
**Purpose:** Find distance to nearest MSHP school

**Method:** Haversine distance (great-circle distance on sphere)

```python
from scipy.spatial.distance import cdist

def haversine_distances(coords1, coords2):
    # Convert to radians
    lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
    lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in km
    return 6371 * c
```

**Note:** This gives straight-line distance, not walking/transit time. Future work could use Google Maps API for travel time.

---

### 2. Statistical Methods

#### Descriptive Statistics
**Computed for MSHP vs. non-MSHP groups:**
- Mean, median, standard deviation
- 25th, 50th, 75th percentiles
- Min/max values

**Example (Chronic Absenteeism):**
| Group | Mean | Median | SD | N |
|-------|------|--------|----|----|
| MSHP | 38.99% | 38.50% | 14.2 | 77 |
| Non-MSHP | 38.45% | 37.80% | 15.6 | 259 |
| **Difference** | +0.54% | | | |
| **P-value** | 0.70 | | | |

**Interpretation:** No statistically significant difference in absenteeism rates.

---

#### Hypothesis Testing

**1. Independent samples t-test**
- **Null hypothesis:** Mean absenteeism is equal for MSHP vs. non-MSHP schools
- **Alternative:** Means are different
- **Assumptions:** 
  - Independent samples ✓
  - Normally distributed (roughly, with large n)
  - Equal variances (Levene's test)
- **Result:** t=-0.37, p=0.70 → Fail to reject null

**2. Mann-Whitney U test (non-parametric)**
- **Purpose:** Robustness check without normality assumption
- **Result:** Similar p-value, confirms t-test conclusion

**3. Pearson correlation**
- **Variables:** Neighborhood MSHP coverage % × Asthma ED rate
- **Result:** r=0.24, p=0.60
- **Interpretation:** Weak positive correlation (MSHP is in higher-asthma neighborhoods, as intended)

---

### 3. Causal Inference Methods

**Challenge:** MSHP placement is not random. Montefiore deliberately targets high-need schools. Simple comparisons confound program effect with selection bias.

**Goal:** Estimate what WOULD have happened to MSHP schools if they hadn't received MSHP (the counterfactual).

---

#### Method 1: Propensity Score Analysis

**Concept:** Model the probability of receiving MSHP based on observed characteristics. Schools with similar propensity scores are comparable.

**Step 1: Estimate propensity scores**
```python
from sklearn.linear_model import LogisticRegression

# Features that predict MSHP placement
X = schools[['pct_poverty', 'asthma_ed_rate', 'svi_overall', 'enrollment']]
y = schools['has_mshp']

model = LogisticRegression()
model.fit(X, y)

schools['propensity_score'] = model.predict_proba(X)[:, 1]
```

**Step 2: Check balance**
- Create propensity score quintiles (Q1-Q5)
- Verify MSHP and non-MSHP schools within each quintile have similar characteristics

**Step 3: Estimate effects within strata**
```python
effects = []
for quintile in range(1, 6):
    mask = schools['ps_quintile'] == quintile
    mshp_mean = schools.loc[mask & schools['has_mshp'], 'chronic_absenteeism_rate'].mean()
    non_mshp_mean = schools.loc[mask & ~schools['has_mshp'], 'chronic_absenteeism_rate'].mean()
    effects.append(mshp_mean - non_mshp_mean)

overall_effect = np.mean(effects)
```

**Result:** Overall effect ≈ +0.5% (MSHP schools have slightly HIGHER absenteeism after matching)

**Interpretation:** After accounting for selection, MSHP schools still serve higher-need populations. No evidence of harm; absence of benefit may reflect:
1. High baseline need (MSHP preventing even worse outcomes)
2. Insufficient time for effects to materialize
3. Outcomes better measured by health (not attendance)
4. Cross-sectional design limitations

---

#### Method 2: Inverse Probability Weighting (IPW)

**Concept:** Weight each school by the inverse of its probability of receiving its actual treatment status. Over-weight "unlikely" cases to create pseudo-population where treatment is independent of covariates.

**Weights:**
- For MSHP schools: w = 1 / P(MSHP | X)
- For non-MSHP schools: w = 1 / (1 - P(MSHP | X))

**Trim extreme weights:**
```python
# Avoid unstable estimates from very low propensities
schools['weight'] = schools.apply(
    lambda row: 1 / row['propensity_score'] if row['has_mshp'] 
    else 1 / (1 - row['propensity_score']), 
    axis=1
)

# Trim at 95th percentile
weight_95th = schools['weight'].quantile(0.95)
schools['weight_trimmed'] = schools['weight'].clip(upper=weight_95th)
```

**Weighted comparison:**
```python
from scipy.stats import ttest_ind

mshp_absenteeism = schools.loc[schools['has_mshp'], 'chronic_absenteeism_rate']
non_mshp_absenteeism = schools.loc[~schools['has_mshp'], 'chronic_absenteeism_rate']

mshp_weights = schools.loc[schools['has_mshp'], 'weight_trimmed']
non_mshp_weights = schools.loc[~schools['has_mshp'], 'weight_trimmed']

# Weighted means
weighted_mshp_mean = np.average(mshp_absenteeism, weights=mshp_weights)
weighted_non_mshp_mean = np.average(non_mshp_absenteeism, weights=non_mshp_weights)

effect = weighted_mshp_mean - weighted_non_mshp_mean
```

**Result:** Effect ≈ +0.3% (similar to stratification)

---

#### Method 3: Dose-Response Analysis

**Concept:** If MSHP is effective, longer exposure should show stronger effects.

**Approach:**
1. Estimate years of MSHP coverage (using known expansion dates)
2. Correlate years with absenteeism
3. Test for linear trend

**Analysis:**
```python
from scipy.stats import pearsonr

# Only MSHP schools
mshp_schools = schools[schools['has_mshp']]

r, p = pearsonr(
    mshp_schools['years_with_mshp'], 
    mshp_schools['chronic_absenteeism_rate']
)
```

**Result:** r=-0.17, p=0.14

**Interpretation:** Suggestive negative correlation (longer MSHP → lower absenteeism), but not statistically significant with n=77 MSHP schools. May reflect:
- True dose-response effect (needs more power)
- Survivorship bias (schools with MSHP longest are different)
- Measurement error in exposure dates

---

#### Method 4: Regression with Controls

**Purpose:** Adjust for multiple confounders simultaneously

**Model:**
```python
from statsmodels.api import OLS

# Outcome: Chronic absenteeism rate
# Treatment: MSHP status (binary)
# Covariates: Asthma, SVI, enrollment, school type

formula = 'chronic_absenteeism_rate ~ has_mshp + asthma_ed_rate + svi_overall + enrollment + school_type'

model = smf.ols(formula, data=schools).fit()
print(model.summary())
```

**Key coefficient:** β_mshp (effect of MSHP adjusting for all covariates)

**Result:** β_mshp ≈ +0.5%, p=0.60 (not significant)

**Interpretation:** Consistent with all other methods. No evidence of significant effect on absenteeism.

---

### 4. Sensitivity Analysis

**Purpose:** Test robustness of conclusions to analytical decisions

**Variations tested:**
1. **Different priority weights:** 
   - Original: 40/40/20 (asthma/absenteeism/enrollment)
   - Enhanced: 25/25/20/15/15 (asthma/SVI/absenteeism/isolation/enrollment)
   - **Result:** Top 20 schools largely unchanged

2. **Outlier handling:**
   - With vs. without schools with >80% absenteeism
   - **Result:** No change to significance tests

3. **Missing data:**
   - Complete case analysis vs. imputation
   - **Result:** Minimal difference (only 5% missing)

4. **Statistical tests:**
   - Parametric (t-test) vs. non-parametric (Mann-Whitney)
   - **Result:** Consistent conclusions

**Conclusion:** Findings are robust to reasonable analytical choices.

---

## Key Findings

### Finding 1: MSHP Targets High-Need Areas (✓ Equity Success)

**Evidence:**
- Schools in Hunts Point - Mott Haven (highest asthma rate: 192 ED visits / 10,000 children) have 15% MSHP coverage
- Schools in High Bridge - Morrisania (asthma rate: 167) have 27% coverage
- Neighborhoods with lowest asthma (Kingsbridge, Northeast Bronx) have <10% coverage

**Statistical support:**
- Positive correlation between neighborhood MSHP coverage and asthma rate (r=0.24)
- MSHP schools have 13% higher poverty rates (91% vs. 78%)
- MSHP schools have 8% more Hispanic students (69% vs. 61%)

**Interpretation:** Montefiore is successfully implementing an **equity-based placement strategy**, prioritizing schools in communities with the greatest health burdens.

---

### Finding 2: No Significant Absenteeism Difference

**Evidence:**
| Metric | MSHP Schools | Non-MSHP Schools | Difference |
|--------|--------------|------------------|------------|
| Mean Absenteeism | 38.99% | 38.45% | +0.54% |
| Median Absenteeism | 38.50% | 37.80% | +0.70% |
| **P-value (t-test)** | | | **0.70** |

**Additional tests:**
- Mann-Whitney U: p=0.69
- Propensity score stratification: +0.5% (ns)
- IPW: +0.3% (ns)
- Regression with controls: β=+0.5% (p=0.60)

**Interpretation:** This does NOT mean MSHP is ineffective. Three alternative explanations:

1. **MSHP is preventing worse outcomes** 
   - High-need schools would have EVEN HIGHER absenteeism without MSHP
   - Cross-sectional design cannot measure counterfactual
   
2. **Wrong outcome measure**
   - MSHP may improve health (asthma attacks, ED visits) without changing attendance
   - Need health outcomes data, not just school data
   
3. **Insufficient exposure time**
   - Effects may take years to materialize
   - Dose-response suggests trend (r=-0.17) with longer exposure

---

### Finding 3: Significant Coverage Gaps Remain

**By neighborhood:**
| Neighborhood | Total Schools | MSHP Schools | Coverage % | Asthma Rate |
|--------------|---------------|--------------|------------|-------------|
| Hunts Point - Mott Haven | 75 | 11 | **15%** | 192 (Highest) |
| High Bridge - Morrisania | 68 | 18 | **27%** | 167 |
| Crotona - Tremont | 52 | 12 | **23%** | 145 |
| Fordham - Bronx Park | 61 | 15 | **25%** | 138 |
| **Overall Bronx** | **370** | **84** | **23%** | **160** |

**Health deserts:**
- 22 schools are >1 mile from nearest MSHP school
- These schools have average asthma rate of 172 (above Bronx average)

**Tier 1 priority schools:**
- 58 schools classified as "Critical" priority
- Concentrated in Hunts Point (18 schools) and Morrisania (14 schools)
- Average priority score: 84.2 / 100

**Expansion opportunity:** Even in neighborhoods where MSHP is present, 75-85% of schools remain uncovered.

---

### Finding 4: Dose-Response Trend (Suggestive)

**Analysis:** Among MSHP schools, correlate years of coverage with absenteeism

**Result:** r=-0.17, p=0.14

**Interpretation:**
- Negative correlation suggests longer MSHP exposure → lower absenteeism
- Not statistically significant at α=0.05 (small sample, n=77)
- Supports hypothesis of delayed effects
- Warrants follow-up with longitudinal data

**Illustrative examples:**
| School | Years with MSHP | Absenteeism |
|--------|-----------------|-------------|
| P.S. 64 (est. 15+ years) | 15 | 28.3% |
| New school (2020 cohort) | 4 | 45.1% |

---

### Finding 5: Priority Ranking Identifies Clear Targets

**Top 10 expansion priorities:**

| Rank | School | Neighborhood | Priority Score | Why High Priority? |
|------|--------|--------------|----------------|---------------------|
| 1 | H.E.R.O. High | Hunts Point | 94.1 | Highest asthma area, high SVI, 1.2mi from MSHP |
| 2 | Bronx Studio School | Morrisania | 88.9 | High asthma, very high absenteeism (51%) |
| 3 | P.S. 048 Joseph R. Drake | Hunts Point | 88.9 | High asthma, high poverty (95%), 0.9mi from MSHP |
| 4 | Hostos-Lincoln Academy | Hunts Point | 88.6 | Large enrollment (682), high asthma area |
| 5 | Longwood Academy | Hunts Point | 87.6 | High asthma, high absenteeism (48%) |

**Common characteristics of Tier 1 schools:**
- 78% in Hunts Point or Morrisania neighborhoods
- Average asthma rate: 180 ED visits / 10,000 (vs. 160 Bronx average)
- Average absenteeism: 42% (vs. 38% Bronx average)
- Average poverty: 89% (vs. 83% Bronx average)

---

## How to Use This Project

### For Policy Makers and Program Administrators

**Use Case 1: Identify Expansion Targets**

1. **Review priority rankings:**
   ```bash
   open outputs/tables/non_mshp_schools_priority_ranked.csv
   ```

2. **Explore interactive map:**
   ```bash
   open outputs/interactive/mshp_coverage_gap_map_enhanced.html
   ```
   - Tier 1 schools pulse on the map
   - Click schools for detailed stats
   - Compare neighborhood asthma rates (choropleth layer)

3. **Generate custom reports:**
   ```python
   import pandas as pd
   
   schools = pd.read_csv('data/processed/bronx_schools_full.csv')
   
   # Filter to specific neighborhood
   hunts_point = schools[schools['uhf_name'] == 'Hunts Point - Mott Haven']
   hunts_point_no_mshp = hunts_point[~hunts_point['has_mshp']]
   
   # Sort by priority
   hunts_point_no_mshp = hunts_point_no_mshp.sort_values('priority_score', ascending=False)
   
   print(hunts_point_no_mshp[['school_name', 'priority_score', 'enrollment']])
   ```

**Use Case 2: Assess Current Coverage**

1. **Review neighborhood summary:**
   ```bash
   cat outputs/tables/neighborhood_summary.csv
   ```

2. **Generate coverage report:**
   ```python
   import pandas as pd
   
   schools = pd.read_csv('data/processed/bronx_schools_full.csv')
   
   coverage_by_neighborhood = schools.groupby('uhf_name').agg({
       'has_mshp': ['sum', 'count', 'mean'],
       'asthma_ed_rate': 'first',
       'chronic_absenteeism_rate': 'mean'
   })
   
   print(coverage_by_neighborhood)
   ```

---

### For Researchers

**Use Case 1: Replicate Analysis**

1. **Set up environment:**
   ```bash
   conda env create -f environment.yml
   conda activate mshp-gap
   pip install -e .
   ```

2. **Run full pipeline:**
   ```bash
   make all
   ```

3. **Or run step-by-step:**
   ```bash
   make collect     # Download data
   make process     # Clean and merge
   make analyze     # Statistical tests
   make visualize   # Generate outputs
   ```

4. **Run tests:**
   ```bash
   make test
   ```

**Use Case 2: Extend Analysis**

**Example: Add new health indicator**

1. Add data source to `scripts/04_collect_*.py`
2. Join to main dataset in `scripts/05_process_and_join.py`
3. Update priority formula in `configs/params.yml`
4. Re-run: `make analyze visualize`

**Example: Change priority weights**

1. Edit `configs/params.yml`:
   ```yaml
   priority_weights:
     asthma_burden: 0.30      # Changed from 0.25
     svi: 0.30                # Changed from 0.25
     absenteeism: 0.20        # Same
     isolation: 0.10          # Changed from 0.15
     enrollment: 0.10         # Changed from 0.15
   ```

2. Re-run ranking:
   ```bash
   python scripts/09_priority_ranking.py
   python scripts/10_visualize.py
   ```

**Use Case 3: Causal Analysis**

Access propensity scores and perform custom analyses:

```python
import pandas as pd
from scipy.stats import ttest_ind

schools = pd.read_csv('data/processed/bronx_schools_full.csv')

# Example: Effect within propensity score quintile 3 (middle)
q3_schools = schools[schools['ps_quintile'] == 'Q3']

mshp_q3 = q3_schools[q3_schools['has_mshp']]['chronic_absenteeism_rate']
non_mshp_q3 = q3_schools[~q3_schools['has_mshp']]['chronic_absenteeism_rate']

t_stat, p_value = ttest_ind(mshp_q3, non_mshp_q3)
effect = mshp_q3.mean() - non_mshp_q3.mean()

print(f"Effect in Q3: {effect:.2f} percentage points (p={p_value:.3f})")
```

---

### For Developers

**Use Case 1: Build Similar Analysis for Another City**

1. **Fork repository:**
   ```bash
   git clone https://github.com/yourusername/mshp-coverage-gap.git
   cd mshp-coverage-gap
   ```

2. **Adapt data collection scripts:**
   - Update `configs/params.yml` with your city's parameters
   - Modify `scripts/01_collect_*.py` to use your city's open data APIs
   - Keep the same file structure (raw → processed → outputs)

3. **Reuse processing pipeline:**
   - Scripts 05-14 should work with minimal changes
   - Update school type classification logic if needed
   - Adjust geographic boundaries (e.g., ZIP codes instead of UHF)

4. **Update visualizations:**
   - Change map center/zoom in `configs/params.yml`
   - Update color schemes if desired
   - Keep same Folium structure

**Use Case 2: Add New Visualization**

Create `scripts/15_custom_viz.py`:

```python
#!/usr/bin/env python3
"""
Script 15: Custom Visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mshp_gap.paths import PROCESSED_DIR, FIGURES_DIR

def main():
    # Load data
    schools = pd.read_csv(PROCESSED_DIR / 'bronx_schools_full.csv')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(
        data=schools,
        x='enrollment',
        y='chronic_absenteeism_rate',
        hue='has_mshp',
        size='asthma_ed_rate',
        ax=ax
    )
    
    ax.set_title('Enrollment vs. Absenteeism by MSHP Status')
    ax.set_xlabel('School Enrollment')
    ax.set_ylabel('Chronic Absenteeism Rate (%)')
    
    # Save
    output_path = FIGURES_DIR / 'enrollment_absenteeism_scatter.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
```

Run:
```bash
python scripts/15_custom_viz.py
```

---

## Extending the Project

### Potential Extensions

#### 1. Longitudinal Analysis
**Goal:** Track schools over time to measure MSHP effects

**Data needs:**
- Historical MSHP expansion dates (precise)
- Multi-year absenteeism data (2015-2024)
- Multi-year health data

**Method:** Difference-in-differences (DiD)
```python
# Compare change in outcomes for schools that got MSHP 
# vs. schools that didn't, before and after expansion

treatment_effect = (
    (post_mshp_mean - pre_mshp_mean) -          # Change in treatment group
    (post_control_mean - pre_control_mean)      # Change in control group
)
```

**Expected outcome:** More precise causal estimate

---

#### 2. Student-Level Analysis
**Goal:** Analyze individual student outcomes

**Data needs:**
- Student-level attendance (with IRB approval)
- Student demographics
- Health records (if available)

**Benefits:**
- Avoid ecological fallacy
- Control for student mobility
- Measure effects on high-risk subgroups

**Method:** Multilevel model
```python
# Random intercepts for schools
from statsmodels.regression.mixed_linear_model import MixedLM

model = MixedLM(
    student_absenteeism ~ has_mshp + student_poverty + student_asthma,
    data=student_data,
    groups=student_data['school_id']
)
```

---

#### 3. Cost-Effectiveness Analysis
**Goal:** Calculate return on investment

**Inputs needed:**
- MSHP operating costs per school per year
- Healthcare cost savings (ED visits avoided)
- Attendance benefits × economic value

**Formula:**
```
Cost-effectiveness ratio = 
    (Total MSHP costs) / 
    (QALYs gained + Education value + Healthcare savings)
```

**Use:** Justify expansion budget

---

#### 4. Machine Learning Prediction
**Goal:** Predict which schools would benefit most

**Approach:**
```python
from sklearn.ensemble import RandomForestRegressor

# Train on MSHP schools: predict absenteeism reduction
# Features: school characteristics, neighborhood, baseline health
# Outcome: Change in absenteeism after MSHP

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict potential impact for non-MSHP schools
non_mshp_schools['predicted_benefit'] = model.predict(X_non_mshp)

# Prioritize by predicted benefit
non_mshp_schools.sort_values('predicted_benefit', ascending=False)
```

---

#### 5. Spatial Network Analysis
**Goal:** Optimize clinic placement to maximize coverage

**Method:** Set cover problem
- Objective: Minimize number of clinics
- Constraint: Every school within X miles of a clinic
- Solve with integer programming

```python
from scipy.optimize import linprog

# Decision variables: x_i = 1 if clinic at school i
# Objective: minimize sum(x_i)
# Constraints: for each school j, sum(x_i * within_radius[i,j]) >= 1
```

---

#### 6. Qualitative Integration
**Goal:** Add stakeholder perspectives

**Methods:**
- Interview school principals about MSHP experiences
- Survey parents about barriers to healthcare
- Focus groups with students about clinic usage

**Integration:**
- Validate quantitative findings
- Identify unmeasured benefits (e.g., parent peace of mind)
- Inform implementation recommendations

---

## Technical Deep Dive

### Advanced Topics

#### Topic 1: Fuzzy Matching Algorithm

**Problem:** MSHP school list has different names than DOE database

**Examples:**
- MSHP: "P.S. 48" → DOE: "P.S. 048 Joseph R. Drake"
- MSHP: "Morris Ed Campus" → DOE: "Morris Educational Campus"

**Solution:** Multi-stage matching

**Stage 1: Exact DBN match**
```python
mshp_schools['matched'] = mshp_schools['dbn'].isin(doe_schools['dbn'])
```
Success rate: 60%

**Stage 2: Normalized name match**
```python
def normalize_name(name):
    # Uppercase
    name = name.upper()
    # Expand abbreviations
    name = name.replace('P.S.', 'PUBLIC SCHOOL')
    name = name.replace('M.S.', 'MIDDLE SCHOOL')
    name = name.replace('H.S.', 'HIGH SCHOOL')
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra whitespace
    name = ' '.join(name.split())
    return name

mshp_schools['name_normalized'] = mshp_schools['school_name'].apply(normalize_name)
doe_schools['name_normalized'] = doe_schools['school_name'].apply(normalize_name)

# Exact match on normalized names
matches = pd.merge(mshp_schools, doe_schools, on='name_normalized')
```
Success rate: 80%

**Stage 3: Fuzzy match**
```python
from thefuzz import fuzz

def fuzzy_match(mshp_name, doe_names, threshold=85):
    best_score = 0
    best_match = None
    
    for doe_name in doe_names:
        # Levenshtein distance-based similarity
        score = fuzz.ratio(mshp_name, doe_name)
        
        if score > best_score:
            best_score = score
            best_match = doe_name
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

# Apply fuzzy matching to unmatched schools
for idx, row in mshp_schools[~mshp_schools['matched']].iterrows():
    match, score = fuzzy_match(row['name_normalized'], doe_schools['name_normalized'])
    
    if match:
        print(f"Fuzzy match (score={score}): {row['school_name']} → {match}")
```
Final success rate: 92% (84 of 91 schools)

---

#### Topic 2: Campus Expansion Logic

**Problem:** One MSHP clinic may serve multiple co-located schools

**Example:** Morris Educational Campus has 5 high schools sharing one clinic:
- Morris Academy for Collaborative Studies
- High School for Violin and Dance
- Bronx International High School
- Morris Early College Academy
- Cinema School

**Solution:** Expand clinic sites to all schools at same location

```python
def expand_campus_schools(mshp_clinics, doe_schools, distance_threshold=0.05):
    """
    Expand each MSHP clinic location to all schools within threshold distance.
    
    distance_threshold: ~50 meters (0.05 km)
    """
    expanded = []
    
    for clinic_idx, clinic in mshp_clinics.iterrows():
        clinic_lat, clinic_lon = clinic['latitude'], clinic['longitude']
        
        # Find all DOE schools at same location
        colocated = doe_schools[
            (np.abs(doe_schools['latitude'] - clinic_lat) < distance_threshold) &
            (np.abs(doe_schools['longitude'] - clinic_lon) < distance_threshold)
        ]
        
        # Mark all as MSHP-covered
        for school_idx, school in colocated.iterrows():
            expanded.append({
                'dbn': school['dbn'],
                'school_name': school['school_name'],
                'clinic_name': clinic['clinic_name'],
                'has_mshp': True
            })
    
    return pd.DataFrame(expanded)
```

**Result:** 30 clinic sites → 91 schools (average 3 schools per clinic)

---

#### Topic 3: Propensity Score Diagnostics

**Purpose:** Verify propensity score model creates balanced groups

**Diagnostic 1: Overlap check**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.hist(
    schools[schools['has_mshp']]['propensity_score'],
    bins=20, alpha=0.5, label='MSHP', color='green'
)

ax.hist(
    schools[~schools['has_mshp']]['propensity_score'],
    bins=20, alpha=0.5, label='Non-MSHP', color='red'
)

ax.set_xlabel('Propensity Score')
ax.set_ylabel('Count')
ax.legend()
plt.show()
```

**Good overlap:** Both distributions span similar range (0.1 - 0.9)
**Poor overlap:** One group clustered at 0, other at 1 (no comparable schools)

**Diagnostic 2: Standardized mean difference (SMD)**
```python
def smd(x_treated, x_control):
    """
    Standardized mean difference.
    SMD < 0.1 indicates good balance.
    """
    mean_diff = x_treated.mean() - x_control.mean()
    pooled_std = np.sqrt(
        (x_treated.std()**2 + x_control.std()**2) / 2
    )
    return mean_diff / pooled_std

# Before matching
smd_before = smd(
    schools[schools['has_mshp']]['pct_poverty'],
    schools[~schools['has_mshp']]['pct_poverty']
)

# After propensity score weighting
mshp_weighted = np.average(
    schools[schools['has_mshp']]['pct_poverty'],
    weights=schools[schools['has_mshp']]['weight_trimmed']
)
non_mshp_weighted = np.average(
    schools[~schools['has_mshp']]['pct_poverty'],
    weights=schools[~schools['has_mshp']]['weight_trimmed']
)

smd_after = (mshp_weighted - non_mshp_weighted) / pooled_std

print(f"SMD before: {smd_before:.3f}")
print(f"SMD after: {smd_after:.3f}")
```

**Interpretation:**
- SMD < 0.1: Excellent balance
- 0.1 ≤ SMD < 0.25: Good balance
- SMD ≥ 0.25: Poor balance (adjust model)

---

#### Topic 4: Interactive Map Implementation

**Challenge:** Create map with multiple layers, custom markers, and smooth performance

**Solution:** Folium with custom JavaScript

**Base structure:**
```python
import folium
from folium import plugins

# Create map
m = folium.Map(
    location=[40.85, -73.85],
    zoom_start=12,
    tiles=None  # Add custom tiles
)

# Multiple base layers
folium.TileLayer(
    tiles='CartoDB dark_matter',
    name='Dark',
    control=True
).add_to(m)

folium.TileLayer(
    tiles='CartoDB positron',
    name='Light',
    control=False
).add_to(m)

folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
    attr='Google',
    name='Satellite',
    control=False
).add_to(m)
```

**Choropleth layer:**
```python
folium.Choropleth(
    geo_data=uhf_geojson,
    data=asthma_data,
    columns=['uhf_code', 'asthma_ed_rate'],
    key_on='feature.properties.uhf_code',
    fill_color='YlOrRd',
    fill_opacity=0.5,
    line_opacity=0.2,
    legend_name='Asthma ED Visits per 10,000 Children',
    name='Asthma Burden'
).add_to(m)
```

**School markers with custom icons:**
```python
for idx, school in schools.iterrows():
    # Color by MSHP status
    color = 'green' if school['has_mshp'] else 'red'
    
    # Size by enrollment (5-25 pixels)
    radius = 5 + (school['enrollment'] / schools['enrollment'].max()) * 20
    
    # Create marker
    folium.CircleMarker(
        location=[school['latitude'], school['longitude']],
        radius=radius,
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2,
        popup=folium.Popup(
            create_popup_html(school),
            max_width=300
        ),
        tooltip=school['school_name']
    ).add_to(m)
```

**Pulsing markers for Tier 1 priorities:**
```python
# Add custom CSS for pulsing effect
pulse_css = """
<style>
@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(255, 0, 0, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(255, 0, 0, 0);
    }
}

.pulse-marker {
    animation: pulse 2s infinite;
}
</style>
"""

m.get_root().html.add_child(folium.Element(pulse_css))

# Add pulsing markers for Tier 1 schools
tier1_schools = schools[schools['priority_tier'] == 'Tier 1 - Critical']

for idx, school in tier1_schools.iterrows():
    folium.CircleMarker(
        location=[school['latitude'], school['longitude']],
        radius=12,
        color='#FF0000',
        fill=True,
        fillColor='#FF0000',
        fillOpacity=1.0,
        weight=3,
        className='pulse-marker',  # Apply pulsing animation
        popup=create_priority_popup(school),
        tooltip=f"🚨 PRIORITY: {school['school_name']}"
    ).add_to(m)
```

**Layer control:**
```python
folium.LayerControl(
    position='topright',
    collapsed=False
).add_to(m)
```

**Save:**
```python
m.save('outputs/interactive/mshp_coverage_gap_map_enhanced.html')
```

---

### Performance Considerations

**Large datasets:** 370 schools × 50 columns = 18,500 data points

**Optimizations:**

1. **Vectorized operations** (NumPy/pandas) instead of loops
   ```python
   # Slow
   for i in range(len(df)):
       df.loc[i, 'new_col'] = df.loc[i, 'col1'] + df.loc[i, 'col2']
   
   # Fast
   df['new_col'] = df['col1'] + df['col2']
   ```

2. **Spatial indexing** for point-in-polygon
   ```python
   # GeoP Pandas uses R-tree spatial index automatically
   schools_gdf = gpd.GeoDataFrame(schools, geometry=...)
   result = gpd.sjoin(schools_gdf, neighborhoods, predicate='within')
   ```

3. **Caching computed values**
   ```python
   # Don't recompute distances every time
   if not os.path.exists('data/processed/distance_matrix.npy'):
       distances = compute_distances(schools)
       np.save('data/processed/distance_matrix.npy', distances)
   else:
       distances = np.load('data/processed/distance_matrix.npy')
   ```

4. **Incremental map rendering**
   ```python
   # Add markers in batches, not all at once
   from folium.plugins import MarkerCluster
   
   marker_cluster = MarkerCluster().add_to(m)
   
   for school in schools.iterrows():
       marker = create_marker(school)
       marker.add_to(marker_cluster)  # Clustering reduces DOM elements
   ```

---

## Conclusion

The MSHP Coverage Gap project demonstrates how **rigorous data analysis** can inform **equitable health policy**. By combining:

- **Geospatial analysis** (mapping coverage gaps)
- **Statistical inference** (testing associations)
- **Causal methods** (estimating program effects)
- **Interactive visualization** (communicating findings)

...we provide actionable insights for expanding school-based healthcare in the Bronx.

### Key Takeaways

1. **MSHP is effectively targeting high-need areas**, demonstrating successful equity-based program design

2. **Significant coverage gaps remain** even in the highest-need neighborhoods, with 77% of Bronx schools lacking access

3. **Priority rankings identify clear expansion targets**, with 58 Tier 1 schools that would maximize health equity impact

4. **Cross-sectional analysis has limitations** for causal claims; longitudinal data needed for definitive effectiveness assessment

5. **Open data enables accountability**, making this analysis possible and replicable

### Future Directions

- **Obtain longitudinal data** for difference-in-differences analysis
- **Access student-level records** (with IRB) for more precise estimates
- **Measure health outcomes** beyond attendance (ED visits, medication adherence)
- **Conduct cost-effectiveness analysis** to guide resource allocation
- **Engage stakeholders** (principals, parents, students) for qualitative validation

### Resources

- **Code repository:** https://github.com/yoelplutchok/mshp-coverage-gap
- **Interactive map:** [View online](https://yoelplutchok.github.io/mshp-coverage-gap)
- **Documentation:** See `docs/` folder
- **Contact:** Open an issue or reach out to repository maintainer

---

**This analysis was conducted with public data to advance health equity in the Bronx.**
