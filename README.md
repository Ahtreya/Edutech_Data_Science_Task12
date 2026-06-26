# Edutech_Data_Science_Task12
# Edutech Data Science Task 12: Feature Engineering

## 📌 Project Overview
This project focuses on executing core feature engineering techniques on the **Housing Dataset** to optimise structural variables for downstream machine learning modelling.

## 🛠️ Implemented Techniques

### 1. Categorical Encoding
- **Label Encoding**: Processed the ordinal string feature `furnishingstatus` into numeric rankings (`0, 1, 2`).
- **One-Hot Encoding**: Converted nominal categorical columns (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) into structural binary flags using `pd.get_dummies(drop_first=True)` to intentionally prevent the dummy variable trap.

### 2. Interaction Features
Created three logical interaction columns to capture spatial layout densities, using safe zero-clipping (`.clip(lower=1)`) to systematically avoid mathematical division-by-zero exceptions:
- `bed_to_bath_ratio`: Ratio of bedrooms to bathrooms.
- `total_rooms`: Combined sum of bedrooms and bathrooms.
- `area_per_room`: Total structural footprint area divided by total room count.

### 3. Logarithmic Transformation
Addressed high right-skewness across distribution scales using natural log scaling (`np.log1p`):
- **Price Skewness**: Successfully normalised target scale variances.
- **Area Skewness**: Reduced asymmetric trailing variance across localised feature profiles.

## 📊 Final Deliverables
- `Edutech_Data_Science_Task12.ipynb`: Complete executable pipeline code.
- `Housing.csv`: Source data file.
- `EnhancedDataset.csv`: Final preprocessed numerical matrix ready for model consumption.
