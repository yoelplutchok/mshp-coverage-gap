# Data Acquisition Log

## Required Datasets

| File | Source | Status | Date Acquired |
|------|--------|--------|---------------|
| doe_school_locations_2025-12-25.csv | NYC Open Data API (wg9x-4ke6) | Complete | 2025-12-25 |
| mshp_school_list_2025-12-25.csv | Montefiore website (manual) |  Original (30 clinics) | 2025-12-25 |
| mshp_school_list_expanded_2025-12-25.csv | NYC DOE SBHC 2025-2026 | Complete (91 schools) | 2025-12-25 |
| doe_attendance_2025-12-25.csv | NYC DOE InfoHub | Complete | 2025-12-25 |
| asthma_ed_visits_uhf_2025-12-25.csv | NYC DOHMH EHDP GitHub | Complete | 2025-12-25 |
| asthma_hospitalizations_uhf_2025-12-25.csv | NYC DOHMH EHDP GitHub | Complete | 2025-12-25 |
| uhf_42_neighborhoods.geojson | NYC Health coronavirus-data GitHub | Complete | 2025-12-25 |

## File Details

### DOE School Locations
- **File**: `doe_school_locations_2025-12-25.csv`
- **Records**: 473 Bronx schools
- **Source URL**: https://data.cityofnewyork.us/resource/wg9x-4ke6.json
- **Key columns**: location_code, location_name, latitude, longitude, geographical_district_code
- **Filter applied**: Districts 7-12 (Bronx), managed_by_name = 'DOE'

### Chronic Absenteeism
- **File**: `doe_attendance_2025-12-25.csv`
- **Records**: 346 Bronx schools
- **Source URL**: https://infohub.nyced.org (public-school-attendance-results-2019-2025.xlsx)
- **Key columns**: dbn, school_name, year, enrollment, chronic_absenteeism_rate
- **Year**: 2024-25

### Asthma ED Visits
- **File**: `asthma_ed_visits_uhf_2025-12-25.csv`
- **Records**: 42 UHF neighborhoods
- **Source URL**: https://github.com/nychealth/EHDP-data (indicator 2380)
- **Key columns**: uhf_code, uhf_name, rate_per_10000, age_group, year
- **Age group**: Children 5-17
- **Year**: 2022

### Asthma Hospitalizations
- **File**: `asthma_hospitalizations_uhf_2025-12-25.csv`
- **Records**: 42 UHF neighborhoods
- **Source URL**: https://github.com/nychealth/EHDP-data (indicator 2048)
- **Key columns**: uhf_code, uhf_name, rate_per_10000, age_group, year
- **Age group**: Children 5-17
- **Year**: 2022

### UHF 42 Boundaries
- **File**: `../geo/uhf_42_neighborhoods.geojson`
- **Features**: 43 (42 UHF + 1 minor)
- **Bronx neighborhoods**: 7
- **Source URL**: https://github.com/nychealth/coronavirus-data/blob/master/Geography-resources/UHF_resources/UHF42.geo.json
- **Key properties**: GEOCODE, GEONAME, BOROUGH

## Manual Data Collection

### MSHP School List
- **File**: `../manual/mshp_school_list_2025-12-25.csv`
- **Records**: 30 schools
- **Source**: https://montefioreeinstein.org/patient-care/school-health
- **Columns**: clinic_name, school_name, school_address

**RESOLVED**: The discrepancy was explained by the **campus model**:
- Original 30 entries = clinic sites (each campus may contain multiple schools)
- Expanded list = 91 unique schools with DBNs (1 duplicate removed)
- Source: NYC DOE SBHC 2025-2026 official document

**Coverage Verification**:
- 29/30 original clinic sites are covered in expanded list
- 1 missing: M.S. 45 (2502 Lorillard Place) - not in NYC DOE SBHC 2025-2026 document

**Files**:
- `mshp_school_list_2025-12-25.csv` - Original 30 clinic sites from Montefiore website
- `mshp_school_list_expanded_2025-12-25.csv` - Expanded 91 unique schools with DBNs

### MSHP Expanded List Details
- **File**: `../manual/mshp_school_list_expanded_2025-12-25.csv`
- **Records**: 91 unique schools (1 duplicate removed: KIPP Academy listed at 2 campuses)
- **Source**: NYC DOE SBHC 2025-2026 (https://www.schools.nyc.gov/docs/default-source/default-document-library/nyc-school-based-health-centers-sbhcs-2025-2026.pdf)
- **Columns**: clinic_name, school_name, school_dbn, school_address, school_type, source, confidence
- **Documentation**: See `docs/mshp_school_list_notes.md`

## Additional Files Downloaded

| File | Description | Size |
|------|-------------|------|
| public-school-attendance-results-2019-2025.xlsx | Full NYC attendance data (all years) | 88 MB |
| asthma_air_quality_uhf_2025-12-25.csv | PM2.5-attributable asthma data | 212 KB |
| uhf42_lookup.csv | UHF code to name lookup table | 1 KB |
