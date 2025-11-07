import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# -----------------------------
# Setup
# -----------------------------
sns.set(style="whitegrid")
simplefilter("ignore", category=ConvergenceWarning)

DATA_FILE = "Housing.csv"
GDRIVE_ID = "1sHuWYSQSVWWhCx_Q1QMpg8PObhqSDJsN"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

if not os.path.exists(DATA_FILE):
    print("Downloading Housing.csv ...")
    gdown.download(GDRIVE_URL, DATA_FILE, quiet=False)

df = pd.read_csv(DATA_FILE)

print("\n=== Head ===")
print(df.head().to_string(index=False))

# -----------------------------
# Part 3 — EDA
# -----------------------------
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# Categorical count plots
categorical_features_to_plot = ['mainroad', 'airconditioning', 'prefarea', 'furnishingstatus']
plt.figure(figsize=(8, 6))
for i, feature in enumerate(categorical_features_to_plot):
    plt.subplot(2, 2, i + 1)
    order = df[feature].value_counts().index
    sns.countplot(x=feature, data=df, order=order)
    plt.title(f'Count: {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    if feature == 'furnishingstatus':
        plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# Numeric distributions
eda_numerical_features = ['price', 'area', 'bedrooms']
plt.figure(figsize=(12, 4))
for i, feature in enumerate(eda_numerical_features):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[feature], kde=True, bins=25)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
plt.tight_layout()
plt.show()

# Pairplot for numerical features
df_numerical = df[numerical_features]
sns.pairplot(df_numerical)
plt.suptitle('Pairwise Relationships (Numeric)', y=1.02)
plt.show()

# -----------------------------
# Part 4 — Cleaning/Encoding/Scaling
# -----------------------------
# Duplicates
num_duplicates = df.duplicated().sum()
original_shape = df.shape
if num_duplicates > 0:
    df = df.drop_duplicates()
new_shape = df.shape
print(f"\nDuplicates found: {num_duplicates}")
print(f"Shape before: {original_shape}, after: {new_shape}")

# Missing values
missing_values_count = df.isnull().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
if missing_values_count.empty:
    print("No missing values found.")
else:
    print("\nMissing values per column:")
    print(missing_values_count)
    df = df.dropna()
    print("Rows with missing values removed.")

# Outlier removal (IQR) for skewed columns
df = pd.read_csv(DATA_FILE)  # start fresh for a clean pipeline
outlier_features = ['price', 'area']
orig_rows = df.shape[0]
for col in outlier_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
rows_removed = orig_rows - df.shape[0]
print(f"\nOutlier rows removed: {rows_removed}")
print(f"Shape after outlier removal: {df.shape}")

# Binary maps
binary_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_vars] = df[binary_vars].apply(lambda x: x.map({'yes': 1, 'no': 0}))

# One-hot for furnishingstatus
df = pd.concat([df, pd.get_dummies(df['furnishingstatus'], drop_first=True)], axis=1)
df = df.drop(columns=['furnishingstatus'])

print("\n=== Encoded dataset head ===")
print(df.head().to_string(index=False))
print("Final shape:", df.shape)

# Train/Test split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=100
)
print("\nSplit shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# Scale selected numeric columns with MinMax (feature range [0,1])
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
scaler = MinMaxScaler()
X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

# Scale y for some models (keep original y for reporting too)
y_scaler = MinMaxScaler()
y_train_scaled = pd.Series(
    y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
    index=y_train.index, name='price'
)
y_test_scaled = pd.Series(
    y_scaler.transform(y_test.values.reshape(-1, 1)).flatten(),
    index=y_test.index, name='price'
)

print("\nAfter normalization (MinMax) on selected columns:")
print(X_train.head().to_string())

# -----------------------------
# Part 5 — Correlation, PCA (CEV), VIF, RFE
# -----------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 9})
plt.title('Correlation Matrix (Processed Housing Features)')
plt.show()

# PCA cumulative explained variance on X_train
pca_full = PCA(n_components=X_train.shape[1])
pca_full.fit(X_train)
cev = np.cumsum(pca_full.explained_variance_ratio_)
components_for_90 = np.argmax(cev >= 0.90) + 1
components_for_95 = np.argmax(cev >= 0.95) + 1

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cev) + 1), cev, marker='o', linestyle='--')
plt.axhline(0.90, color='r', label='90%')
plt.axhline(0.95, color='g', label='95%')
plt.axvline(components_for_90, color='r', linestyle='--')
plt.axvline(components_for_95, color='g', linestyle='--')
plt.title('Cumulative Explained Variance by PCs')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, X_train.shape[1] + 1))
plt.legend()
plt.grid(True)
plt.show()

print("\nPCA CEV:")
print(f"Total PCs: {X_train.shape[1]}")
print(f"PCs for 90%: {components_for_90}")
print(f"PCs for 95%: {components_for_95}")

# VIF (drop until all < 5)
def calculate_vif(df_in: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame({"Feature": df_in.columns})
    df_vif_const = df_in.assign(const=1).astype(float)
    vif_data["VIF"] = [
        variance_inflation_factor(df_vif_const.values, i)
        for i in range(df_vif_const.shape[1] - 1)
    ]
    return vif_data.sort_values(by="VIF", ascending=False)

X_train_vif = X_train.copy()
vif_df = calculate_vif(X_train_vif)
while vif_df['VIF'].max() > 5 and X_train_vif.shape[1] > 1:
    drop_feat = vif_df.iloc[0]['Feature']
    X_train_vif = X_train_vif.drop(columns=[drop_feat])
    vif_df = calculate_vif(X_train_vif)

print("\nVIF after pruning (target < 5):")
print(vif_df.to_string(index=False))
print(f"Final #features after VIF: {X_train_vif.shape[1]} (from {X_train.shape[1]})")

# RFE with Linear Regression (select 8 features)
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=8)
rfe.fit(X_train, y_train)

rfe_results = pd.DataFrame({
    'Features': X_train.columns,
    'RFE_Support': rfe.support_,
    'Ranking': rfe.ranking_
}).sort_values(by='Ranking')

selected_features_rfe = rfe_results[rfe_results['RFE_Support'] == True]['Features'].tolist()
print("\nRFE selected features (n=8):", selected_features_rfe)
print("\nRFE ranking table:")
print(rfe_results.to_string(index=False))

# -----------------------------
# Part 6 — Modeling
# -----------------------------
# Linear Regression with PCA(10)
pca10 = PCA(n_components=10, random_state=100)
X_train_pca10 = pca10.fit_transform(X_train)
X_test_pca10 = pca10.transform(X_test)
lr_pca = LinearRegression().fit(X_train_pca10, y_train)
y_pred_test_lr_pca = lr_pca.predict(X_test_pca10)
print("\nLinear Regression + PCA(10)")
print(f"R² (Train): {r2_score(y_train, lr_pca.predict(X_train_pca10)):.4f}")
print(f"R² (Test):  {r2_score(y_test, y_pred_test_lr_pca):.4f}")
print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test_lr_pca):,.2f}")

# Ridge with PCA(10)
alphas = np.logspace(-4, 0, 10)
ridge_pca = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error').fit(X_train_pca10, y_train)
y_pred_test_ridge = ridge_pca.predict(X_test_pca10)
print("\nRidge + PCA(10)")
print(f"Best alpha: {ridge_pca.alpha_:.4f}")
print(f"R² (Train): {r2_score(y_train, ridge_pca.predict(X_train_pca10)):.4f}")
print(f"R² (Test):  {r2_score(y_test, y_pred_test_ridge):.4f}")
print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test_ridge):,.2f}")

# Lasso with PCA(10)
alphas = np.logspace(-4, 0, 100)
lasso_pca = LassoCV(alphas=alphas, cv=5, random_state=100, max_iter=10000).fit(X_train_pca10, y_train)
y_pred_test_lasso = lasso_pca.predict(X_test_pca10)
print("\nLasso + PCA(10)")
print(f"Best alpha: {lasso_pca.alpha_:.4f}")
print(f"R² (Train): {r2_score(y_train, lasso_pca.predict(X_train_pca10)):.4f}")
print(f"R² (Test):  {r2_score(y_test, y_pred_test_lasso):.4f}")
print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test_lasso):,.2f}")

# Polynomial (degree=2) + PCA(95% variance) + Linear Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
pca95 = PCA(n_components=0.95, random_state=100)
X_train_poly_pca = pca95.fit_transform(X_train_poly)
X_test_poly_pca = pca95.transform(X_test_poly)
model_poly_pca = LinearRegression().fit(X_train_poly_pca, y_train)
y_pred_test_poly = model_poly_pca.predict(X_test_poly_pca)
print("\nPolynomial (deg=2) + PCA(95%) + LR")
print(f"R² (Train): {r2_score(y_train, model_poly_pca.predict(X_train_poly_pca)):.4f}")
print(f"R² (Test):  {r2_score(y_test, y_pred_test_poly):.4f}")

print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_test_poly):,.2f}")

# MLPRegressor with scaling + PCA(95%)
scaler_X = StandardScaler()
scaler_y_std = StandardScaler()
X_train_std = scaler_X.fit_transform(X_train)
X_test_std = scaler_X.transform(X_test)
y_train_std = scaler_y_std.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_std = scaler_y_std.transform(y_test.values.reshape(-1, 1)).ravel()

pca_mlp = PCA(n_components=0.95, random_state=100)
X_train_std_pca = pca_mlp.fit_transform(X_train_std)
X_test_std_pca = pca_mlp.transform(X_test_std)

mlp_pca = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='tanh',
    solver='adam',
    learning_rate_init=1e-4,
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42,
    verbose=False
).fit(X_train_std_pca, y_train_std)

y_pred_std = mlp_pca.predict(X_test_std_pca)
y_pred = scaler_y_std.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
print("\nMLPRegressor + PCA(95%)")
print(f"R² (Test):  {r2_score(y_test, y_pred):.4f}")
print(f"MAE (Test): {mean_absolute_error(y_test, y_pred):,.2f}")

# Elastic Net on scaled (no PCA)
elastic = ElasticNetCV(
    l1_ratio=np.linspace(0.05, 0.95, 20),
    alphas=np.logspace(-4, 2, 100),
    cv=10,
    max_iter=20000,
    random_state=42
).fit(X_train_std, y_train_std)
y_pred_en_std = elastic.predict(X_test_std)
y_pred_en = scaler_y_std.inverse_transform(y_pred_en_std.reshape(-1, 1)).ravel()
print("\nElastic Net (scaled)")
print(f"Optimal α: {elastic.alpha_:.5f}")
print(f"Optimal L1 Ratio: {elastic.l1_ratio_:.2f}")
print(f"R² (Test):  {r2_score(y_test, y_pred_en):.4f}")
print(f"MAE (Test): {mean_absolute_error(y_test, y_pred_en):,.2f}")

# -----------------------------
# MLPClassifier feature extraction + regressors
# -----------------------------
# Create 4-bin target on training set
bins = y_train.quantile([0.25, 0.5, 0.75]).tolist()
bins = [y_train.min()] + bins + [y_train.max()]
labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
y_train_binned = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True, duplicates='drop')
le = LabelEncoder()
y_train_class = le.fit_transform(y_train_binned.astype(str))

def extract_features(mlp_model: MLPClassifier, X_df: pd.DataFrame, layer_index: int = -2) -> np.ndarray:
    """
    Extracts activations from the last hidden layer (assumes relu).
    """
    current = X_df.values
    # layer_index=-2 means all hidden layers (exclude final output layer)
    for i in range(len(mlp_model.coefs_) + layer_index):
        z = current @ mlp_model.coefs_[i] + mlp_model.intercepts_[i]
        current = np.maximum(0, z)
    return current

mlp_feat = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-2,
    max_iter=2000,
    early_stopping=True,
    random_state=42
).fit(X_train, y_train_class)

X_train_feat = extract_features(mlp_feat, X_train)
X_test_feat = extract_features(mlp_feat, X_test)

def evaluate_model(model, Xtr, Xte, name, ytr=y_train, yte=y_test):
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    return name, r2_score(yte, yhat), mean_absolute_error(yte, yhat)

results = {}
for mdl, name in [
    (LinearRegression(), "Multiple Linear Regression"),
    (RidgeCV(alphas=np.logspace(-4, 0, 100), cv=5), "Ridge"),
    (LassoCV(alphas=np.logspace(-4, 0, 100), cv=5, random_state=100), "Lasso"),
]:
    mname, r2v, mae = evaluate_model(mdl, X_train_feat, X_test_feat, name)
    results[mname] = {"R2": r2v, "MAE": mae}

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_feat_poly = poly.fit_transform(X_train_feat)
X_test_feat_poly = poly.transform(X_test_feat)
mname, r2v, mae = evaluate_model(LinearRegression(), X_train_feat_poly, X_test_feat_poly, "Polynomial")
results[mname] = {"R2": r2v, "MAE": mae}

comparison_df = pd.DataFrame(results).T.reset_index()
comparison_df.columns = ['Model', 'R²', 'MAE']
print("\nFinal Regression Results on MLP-Extracted Features (sorted by MAE)")
print(comparison_df.sort_values(by='MAE').to_string(index=False, float_format="%.4f"))
