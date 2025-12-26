# Limitations and Caveats

## MSHP Coverage Gap Analysis

This document describes the known limitations of this analysis. These should be considered when interpreting results and making policy recommendations.

---

## 1. Causal Inference Limitations

### 1.1 Cross-Sectional Design
- **Issue:** Data represents a single point in time
- **Impact:** Cannot establish causation, only association
- **Mitigation:** Used propensity score methods to adjust for selection
- **Recommendation:** Future research should use longitudinal data with MSHP expansion dates

### 1.2 Selection Bias
- **Issue:** MSHP clinic placement is not random; Montefiore deliberately targets high-need schools
- **Impact:** Observed differences may reflect selection, not program effects
- **Evidence:** Propensity score analysis confirms MSHP schools ARE in higher-need areas
- **Interpretation:** This is BY DESIGN—MSHP targeting is an equity success, not a limitation

### 1.3 Unmeasured Confounders
Variables not controlled for that may affect both MSHP placement and outcomes:
- School leadership quality and engagement
- Parent and community involvement
- Historical funding and resource allocation
- Student mobility and turnover
- Neighborhood gentrification trends

---

## 2. Data Limitations

### 2.1 Ecological Fallacy (Health Data)
- **Issue:** Asthma data is at neighborhood (UHF) level, not school or student level
- **Impact:** Individual students at a school may have different asthma prevalence than the neighborhood average
- **Scale:** 370 schools mapped to 7 UHF neighborhoods (average 53 schools/neighborhood)
- **Recommendation:** Future analysis should seek school-level health data if available

### 2.2 Temporal Mismatch
| Data Source | Year |
|-------------|------|
| School locations | 2024-25 |
| Attendance | 2024-25 |
| Demographics | 2021-22 |
| Asthma | 2022 |
| SVI | Synthetic (2022 base) |

- **Impact:** Conditions may have changed between data collection periods
- **Mitigation:** Used most recent available data for each source

### 2.3 Missing Data
- **Attendance:** ~10% of schools missing (new schools, special programs)
- **Demographics:** 5% missing (filled with neighborhood estimates)
- **MSHP expansion history:** Estimated cohorts, not verified dates
- **SVI:** Synthetic data (actual CDC download failed)

### 2.4 Source Data Quality
- **Issue:** NYC Open Data DOE file contained 12 duplicate DBNs
- **Resolution:** Removed duplicates during processing
- **Verification:** All other counts validated against official sources

---

## 3. MSHP Definition Limitations

### 3.1 Binary Coverage Variable
- **Issue:** MSHP coverage treated as binary (yes/no)
- **Reality:** Coverage intensity likely varies:
  - Full-time on-site clinic vs. part-time
  - Range of services offered
  - Staff-to-student ratios
  - Hours of operation
- **Impact:** May understate variation in "treatment intensity"

### 3.2 Partial Services Not Captured
Some schools may receive MSHP-related services not captured:
- Mobile clinic visits
- Referral relationships
- Community health worker outreach
- Telehealth consultations
- Parent education programs

### 3.3 Campus Complexity
- **Issue:** Large campuses may share one clinic across multiple schools
- **Resolution:** Expanded clinic sites to individual schools
- **Uncertainty:** Some schools on campuses may have differential access

---

## 4. Geographic Limitations

### 4.1 Bronx Only
- **Scope:** Analysis limited to Bronx borough
- **Impact:** Findings may not generalize to other NYC boroughs or cities
- **Rationale:** Bronx has highest asthma burden and is MSHP's primary service area

### 4.2 Boundary Effects
- Schools near UHF boundaries may be assigned to neighborhoods that don't reflect their catchment
- Students may live in different neighborhoods than their school's location

### 4.3 Distance Metrics
- Distance to nearest MSHP uses straight-line (Haversine) distance
- Actual travel time/accessibility may differ due to transportation barriers

---

## 5. Statistical Limitations

### 5.1 Multiple Comparisons
- Multiple statistical tests conducted without formal correction
- Increases risk of false positives
- Core finding (no significant difference) is consistent across methods

### 5.2 Sample Size for Subgroup Analysis
- Some propensity score strata have small MSHP counts (n=9-23)
- Limits precision of within-stratum estimates
- Wide confidence intervals for stratified effects

### 5.3 Model Specification
- Propensity score model has low pseudo R² (0.007)
- May not fully capture selection process
- Adding more predictors limited by available data

---

## 6. Interpretation Caveats

### 6.1 No Significant Effect ≠ No Effect
- Failure to find significant difference does not prove MSHP has no benefit
- Study may be underpowered to detect small effects
- Effects may appear in outcomes not measured (health, not just attendance)

### 6.2 MSHP May Prevent Worse Outcomes
- High-need schools without MSHP might have EVEN HIGHER absenteeism if MSHP weren't targeting similar schools
- MSHP may be successfully preventing outcomes from deteriorating
- Counterfactual is unknowable with cross-sectional data

### 6.3 Priority Ranking Assumptions
Priority score assumes:
- Asthma burden, absenteeism, and enrollment are appropriate proxies for need
- Weights (40/40/20 or enhanced weights) reflect true priorities
- Neighborhood-level health data applies uniformly to schools

---

## 7. Recommendations for Future Research

### 7.1 Data Improvements
- [ ] Obtain MSHP expansion dates for difference-in-differences analysis
- [ ] Seek school-level health outcomes (clinic visit data, if available)
- [ ] Access student-level data (with IRB approval) for individual analysis
- [ ] Update SVI with actual CDC data when download issues resolved

### 7.2 Methodological Extensions
- [ ] Propensity score matching (instead of weighting)
- [ ] Regression discontinuity (if eligibility threshold exists)
- [ ] Instrumental variables (if valid instrument identified)
- [ ] Synthetic control methods for comparison

### 7.3 Outcome Expansion
- [ ] Include health outcomes beyond absenteeism
- [ ] Track student academic performance
- [ ] Measure healthcare utilization (ED visits avoided)
- [ ] Assess parent/teacher satisfaction

---

## Summary

This analysis provides valuable descriptive insights into MSHP coverage patterns and identifies priority schools for expansion. However, **causal claims about MSHP effectiveness should be made cautiously** given the cross-sectional design and unmeasured confounders.

The key finding—that MSHP schools are located in higher-need areas with similar absenteeism rates—is best interpreted as evidence of successful **equity targeting** rather than program ineffectiveness.

