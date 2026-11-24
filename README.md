# C.A.S.A. — Comparing Apartments for Student Affordability

Machine Learning & Communication Skills project · Bachelor in Applied Mathematics · IE University · Fall 2026
Philippos Alexopoulos · Tamara Shopova · Matisse Aaftink

Students comparing Madrid rental listings need a reference price that accounts for a flat's characteristics. This project predicts the monthly asking rent of a Madrid flat from its listed features and flags listings whose asking rent lies outside a 90% prediction interval around that estimate.

**Final report:** [reports/CASA_Final_Report.pdf](reports/CASA_Final_Report.pdf)

## Main results

Five-fold out-of-fold metrics on 1,840 flats (after removing 44 duplicates and the 205 highest rents):

| Model | MAE (€) | MedAE (€) | MAPE (%) | R² (log rent) |
|---|---|---|---|---|
| Median baseline | 604.81 | 450.00 | 39.99 | −0.013 |
| Ridge regression | 295.80 | 180.34 | 18.13 | 0.761 |
| Gradient boosting | **260.01** | **167.77** | **16.75** | **0.790** |

- Gradient boosting cuts the baseline's mean error by 57.0%.
- Split-conformal 90% prediction intervals cover 92.9% of asking rents on 368 held-out listings.
- 51 listings lie above their interval and 49 below. These flags describe model residuals: they are a reason to look closer at a listing, not proof that it is mispriced.

## How to reproduce

```bash
conda env create -f environment.yml
conda activate MLProjectBAM
python -m src.model.fair_value                         # models, intervals, flags -> src/data/processed/fair_value_labels.csv
python -m src.model.report_assets --output report_out  # every number and figure in the report
```

All randomness uses `random_state = 0`. Hardware, software versions and run times are listed in `src/model/computing_setup.txt`.

## Repository structure

```
src/
  data/
    data_acquire.py            load the workbook, drop duplicates and the 205 highest rents
    preprocess.py              feature lists, imputation, one-hot encoding (fit inside each CV fold)
    viz.ipynb                  exploratory plots
    houses_for_rent_madrid.xlsx   input data (see "Data")
    geo/
      geocode.py               address geocoding with OpenStreetMap Nominatim (cached)
      geocoded_addresses.csv   cached coordinates
      barrios/                 barrio boundaries from the Ayuntamiento de Madrid geoportal
    processed/                 model outputs, incl. fair_value_labels.csv
    scraper/                   Idealista scraper (blocked by the site; not used for the results)
  model/
    fair_value.py              three models, conformal intervals, flags
    report_assets.py           rebuilds every number and figure in the report
    style.py                   shared colours for all figures
    fair_value_results.ipynb   results figures
    fair_value_map.ipynb       barrio maps
    computing_setup.txt        hardware, versions, run times
```

## Data

`src/data/houses_for_rent_madrid.xlsx`: 2,089 Madrid flat listings with rent, size, bedrooms, floor, lift, exterior, property type, district and neighbourhood (column definitions are in the file's "Legend" sheet).

Source: downloaded from Marian Garabana's public repository [Madrid_Rental_Analysis_StreamlitApp](https://github.com/MarianGarabana/Madrid_Rental_Analysis_StreamlitApp) (file `data/Houses for rent in Madrid.xlsx`, renamed here; the two files are byte-identical). According to that repository, the listings were originally collected from Idealista and are shared for academic purposes only. The collection date and licence are not stated. Rents are asking prices, not signed contracts.

Maps use OpenStreetMap geocoding (Nominatim) and barrio boundaries from the Ayuntamiento de Madrid geoportal.
