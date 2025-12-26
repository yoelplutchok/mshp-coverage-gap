# Methodology

## MSHP Coverage Gap Analysis

---

## 1. Research Questions

### Primary Question
Are Bronx public schools WITHOUT Montefiore School Health Program (MSHP) coverage disproportionately located in neighborhoods with the highest childhood asthma burden and chronic absenteeism?

### Secondary Questions
1. What is the geographic distribution of MSHP-covered vs. non-covered schools?
2. Is there a correlation between MSHP coverage and chronic absenteeism?
3. Which non-covered schools should be prioritized for MSHP expansion?
4. What is the causal effect of MSHP on student outcomes?

---

## 2. Data Sources

### 2.1 School Locations
- **Source:** NYC Open Data API (dataset wg9x-4ke6)
- **Coverage:** All Bronx public schools (K-12)
- **Fields:** DBN, name, address, coordinates, grades
- **Filtering:** Open schools only, valid coordinates

### 2.2 MSHP School List
- **Primary Source:** NYC DOE School-Based Health Centers 2025-2026 Official Document
- **Secondary Source:** Montefiore Einstein website
- **Method:** Cross-referenced official DOE SBHC list with Montefiore sponsorship
- **Key Finding:** Many "clinic sites" serve multiple co-located schools (campus model)

### 2.3 Attendance Data
- **Source:** NYC DOE InfoHub (public-school-attendance-results-2019-2025.xlsx)
- **Metric:** Chronic Absenteeism Rate (% students missing 10%+ of days)
- **Year:** 2024-25 academic year

### 2.4 Health Data
- **Source:** NYC DOHMH Environment & Health Data Portal (EHDP)
- **Primary Indicator:** Asthma ED visits per 10,000 children ages 5-17
- **Geography:** UHF 42 Neighborhoods
- **Year:** 2022 (most recent available)

### 2.5 Geographic Boundaries
- **Source:** NYC Health coronavirus-data GitHub repository
- **Format:** GeoJSON
- **Coverage:** 42 UHF neighborhoods citywide, 7 in Bronx

### 2.6 Demographics
- **Source:** NYC Open Data (dataset c7ru-d68s)
- **Fields:** Poverty, ELL, SWD, race/ethnicity
- **Year:** 2021-22 (most recent in API)

---

## 3. Data Processing

### 3.1 School Standardization
1. Filtered to Bronx borough only
2. Removed schools without valid coordinates
3. Standardized school names (uppercase, abbreviation normalization)
4. Removed 12 duplicate DBNs from source data
5. Classified school type based on name patterns

### 3.2 MSHP Matching
1. Primary match: Direct DBN lookup from official SBHC document
2. Expanded campus schools to individual schools (30 clinic sites → 91 schools)
3. Removed 1 duplicate (KIPP Academy at 2 campuses)
4. Final matched: 84 of 91 MSHP schools found in DOE list

### 3.3 Spatial Join
1. Converted school coordinates to point geometries
2. Ensured consistent CRS (EPSG:4326)
3. Performed point-in-polygon join with UHF boundaries
4. All 370 schools successfully assigned to Bronx UHF neighborhoods

### 3.4 Data Integration
1. Joined attendance data on DBN
2. Joined health data on UHF code
3. Joined demographics on DBN (with synthetic fill for missing)
4. Created composite health burden score

---

## 4. Analysis Methods

### 4.1 Descriptive Statistics
- Calculated means, medians, standard deviations by MSHP status
- Aggregated to neighborhood level for coverage gap analysis

### 4.2 Statistical Testing
- **T-test:** Compared mean chronic absenteeism (MSHP vs. non-MSHP)
- **Mann-Whitney U:** Non-parametric alternative for robustness
- **Pearson correlation:** MSHP coverage vs. neighborhood asthma rate
- **Point-biserial correlation:** MSHP status vs. individual school absenteeism

### 4.3 Priority Ranking

#### Original Priority Score
```
priority_score = 0.4 × asthma_percentile + 
                 0.4 × absenteeism_percentile + 
                 0.2 × enrollment_percentile
```

#### Enhanced Priority Score (with SVI)
```
priority_score = 0.25 × asthma_score + 
                 0.25 × svi_score + 
                 0.20 × absenteeism_score + 
                 0.15 × isolation_score + 
                 0.15 × enrollment_score
```

Where:
- `asthma_score`: Neighborhood asthma ED rate (normalized 0-100)
- `svi_score`: Social Vulnerability Index (normalized 0-100)
- `absenteeism_score`: School chronic absenteeism (normalized 0-100)
- `isolation_score`: Distance to nearest MSHP school (normalized 0-100)
- `enrollment_score`: School enrollment size (normalized 0-100)

#### Tier Classification
- **Tier 1 - Critical:** Top 20% priority score
- **Tier 2 - High:** 60th-80th percentile
- **Tier 3 - Moderate:** 40th-60th percentile
- **Tier 4 - Lower:** Bottom 40%

### 4.4 Causal Inference Analysis

#### Method 1: Propensity Score Estimation
Logistic regression to predict MSHP coverage:
```
P(MSHP = 1) = logit(β₀ + β₁×poverty + β₂×asthma + β₃×svi + β₄×enrollment)
```

#### Method 2: Inverse Probability Weighting (IPW)
Weights calculated as:
- For MSHP schools: w = 1 / P(MSHP)
- For non-MSHP schools: w = 1 / (1 - P(MSHP))
- Weights trimmed at 95th percentile to reduce variance

#### Method 3: Stratification
- Divided sample into 5 propensity score quintiles
- Calculated treatment effect within each stratum
- Averaged across strata for overall estimate

#### Method 4: Dose-Response
- Correlated years of MSHP exposure with absenteeism
- Linear regression: Absenteeism ~ Years with MSHP

### 4.5 Regression Analysis
Multiple linear regression with controls:
```
Absenteeism = β₀ + β₁×MSHP + β₂×Asthma + β₃×SVI + β₄×Enrollment + β₅×SchoolType + ε
```

---

## 5. Visualization Methods

### 5.1 Interactive Map
- **Tool:** Folium (Python)
- **Base layers:** CartoDB Dark Matter, Light, Satellite
- **Choropleth:** UHF neighborhoods colored by asthma rate
- **School markers:** Colored by MSHP status, sized by enrollment
- **Special markers:** Tier 1 priority schools highlighted with pulsing effect

### 5.2 Static Charts
- **Tool:** Matplotlib/Seaborn
- **Charts:** Box plots, bar charts, heatmaps, scatter plots
- **Style:** Seaborn whitegrid with custom color palette

---

## 6. Software Environment

### Dependencies
- Python 3.11
- pandas ≥2.0.0
- geopandas ≥0.14.0
- folium ≥0.15.0
- scipy ≥1.11.0
- statsmodels (for regression)
- matplotlib/seaborn (for visualization)

### Reproducibility
- All parameters stored in `configs/params.yml`
- Random seeds set where applicable
- Pipeline orchestrated via Makefile
- Environment specified in `environment.yml`

---

## 7. Validation

### Data Quality Checks
1. Verified school counts against DOE published totals
2. Confirmed MSHP matches against official SBHC document
3. Validated coordinate bounds (all within Bronx)
4. Cross-checked asthma rates against published reports

### Sensitivity Analysis
- Tested different priority score weights
- Compared results with and without synthetic data
- Verified statistical conclusions robust to duplicate removal

---

## 8. Code Repository

All analysis code available in the `scripts/` directory:
- `01_collect_doe_schools.py` - School location download
- `04_collect_uhf_boundaries.py` - Geographic boundaries
- `05_process_and_join.py` - Data cleaning and integration
- `08_analyze.py` - Statistical analysis
- `10_visualize.py` - Basic visualizations
- `10_visualize_enhanced.py` - Enhanced interactive map
- `11_enhanced_analysis.py` - SVI and accessibility analysis
- `12_enhanced_visualizations.py` - Enhanced charts
- `13_add_demographics_health.py` - Demographics and causal inference
- `14_demographics_visualizations.py` - Final visualizations

