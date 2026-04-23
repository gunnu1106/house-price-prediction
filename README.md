# House Price Prediction

ML model to predict house prices using 5 regression algorithms with feature engineering and model comparison.

## Models Used
| Model | Type |
|-------|------|
| Linear Regression | Baseline |
| Ridge Regression | Regularized |
| Lasso Regression | Feature Selection |
| Random Forest | Ensemble |
| Gradient Boosting | Boosting |

## Features Engineered
- Area (sqft) + log transform
- Bedrooms, Bathrooms, Floors
- House Age, Garage Spaces
- Locality encoding
- Derived: total_rooms, is_new

## Tech Stack
`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn`

## Setup
```bash
git clone https://github.com/gunnu1106/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python model.py
```
---
Made by [Gunjan](https://github.com/gunnu1106)
