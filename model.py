"""
House Price Prediction
Author: Gunjan
Description: ML model to predict house prices using multiple regression
             algorithms with feature engineering and model comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6c63ff','#ff6584','#43aa8b','#f59f00','#4ecdc4']

# ── Data Generation ────────────────────────────────────────
def generate_house_data(n=1500):
    np.random.seed(42)
    area        = np.random.randint(500, 5000, n)
    bedrooms    = np.random.randint(1, 7, n)
    bathrooms   = np.random.randint(1, 5, n)
    floors      = np.random.randint(1, 4, n)
    age         = np.random.randint(0, 50, n)
    garage      = np.random.randint(0, 4, n)
    locality    = np.random.choice(['Premium','Good','Average','Budget'], n, p=[0.15,0.30,0.35,0.20])
    loc_mult    = {'Premium':2.5,'Good':1.8,'Average':1.2,'Budget':0.8}

    price = (
        area * 3500 +
        bedrooms * 200000 +
        bathrooms * 150000 +
        garage * 100000 -
        age * 15000 +
        np.array([loc_mult[l] for l in locality]) * 500000 +
        np.random.normal(0, 200000, n)
    ).astype(int)
    price = np.clip(price, 500000, 50000000)

    return pd.DataFrame({
        'area_sqft': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
        'floors': floors, 'age_years': age, 'garage_spaces': garage,
        'locality': locality, 'price': price
    })


# ── Feature Engineering ────────────────────────────────────
def prepare_features(df):
    df = df.copy()
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    df['total_rooms']    = df['bedrooms'] + df['bathrooms']
    df['is_new']         = (df['age_years'] < 5).astype(int)
    df['area_log']       = np.log1p(df['area_sqft'])

    le = LabelEncoder()
    df['locality_enc'] = le.fit_transform(df['locality'])

    features = ['area_sqft','area_log','bedrooms','bathrooms','floors',
                'age_years','garage_spaces','total_rooms','is_new','locality_enc']
    return df, features


# ── Model Training & Evaluation ────────────────────────────
def train_models(df, features):
    df, features = prepare_features(df)
    X, y = df[features], df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Linear Regression':      LinearRegression(),
        'Ridge Regression':       Ridge(alpha=100),
        'Lasso Regression':       Lasso(alpha=1000),
        'Random Forest':          RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting':      GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    print("\n── Model Performance ─────────────────────────")
    print(f"{'Model':<25} {'R²':>8} {'MAE (₹)':>12} {'RMSE (₹)':>12}")
    print("-" * 60)

    for name, model in models.items():
        X_tr = X_train_s if 'Regression' in name else X_train
        X_te = X_test_s  if 'Regression' in name else X_test
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse,
                         'preds': preds, 'X_te': X_te}
        print(f"  {name:<23} {r2:>8.4f} {mae:>12,.0f} {rmse:>12,.0f}")

    best = max(results, key=lambda k: results[k]['r2'])
    print(f"\n  Best Model: {best} (R² = {results[best]['r2']:.4f})")
    return results, y_test, X_test, features, df


def plot_results(results, y_test, df, features):
    import os; os.makedirs('outputs', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('House Price Prediction - Model Analysis', fontsize=16, fontweight='bold')

    # 1. Model R² Comparison
    names = list(results.keys())
    r2s   = [results[n]['r2'] for n in names]
    axes[0,0].barh([n.replace(' Regression','') for n in names], r2s,
                   color=COLORS[:len(names)])
    axes[0,0].set_title('Model R² Score Comparison', fontweight='bold')
    axes[0,0].set_xlabel('R² Score')
    axes[0,0].set_xlim(0, 1)
    for i, v in enumerate(r2s):
        axes[0,0].text(v + 0.01, i, f'{v:.3f}', va='center')

    # 2. Best Model: Actual vs Predicted
    best = max(results, key=lambda k: results[k]['r2'])
    preds = results[best]['preds']
    axes[0,1].scatter(y_test/1e6, preds/1e6, alpha=0.4, color='#6c63ff', s=20)
    mn, mx = y_test.min()/1e6, y_test.max()/1e6
    axes[0,1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0,1].set_title(f'Actual vs Predicted — {best}', fontweight='bold')
    axes[0,1].set_xlabel('Actual Price (₹M)'); axes[0,1].set_ylabel('Predicted Price (₹M)')
    axes[0,1].legend()

    # 3. Feature Importance (RF)
    rf = results['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
    axes[1,0].barh(imp.index, imp.values, color='#43aa8b')
    axes[1,0].set_title('Feature Importance — Random Forest', fontweight='bold')
    axes[1,0].set_xlabel('Importance')

    # 4. Price Distribution by Locality
    df.groupby('locality')['price'].median().sort_values().plot(
        kind='bar', ax=axes[1,1], color=COLORS[:4], edgecolor='white')
    axes[1,1].set_title('Median Price by Locality', fontweight='bold')
    axes[1,1].set_ylabel('Price (₹)')
    axes[1,1].tick_params(axis='x', rotation=15)
    axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))

    plt.tight_layout()
    plt.savefig('outputs/house_price_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/house_price_prediction.png")


if __name__ == '__main__':
    print("Generating dataset...")
    df = generate_house_data()
    print(f"Dataset: {df.shape} | Price range: ₹{df['price'].min():,} – ₹{df['price'].max():,}")
    results, y_test, X_test, features, df_fe = train_models(df, None)
    plot_results(results, y_test, df, features)
    print("\nDone!")
