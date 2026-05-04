import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

print('Starting model training script...')
DATA_PATH = 'ieee_fraud_detection_pipeline_features.csv'
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f'Dataset loaded. Shape: {df.shape}')

# Drop > 80% missing
null_pct_all = df.isnull().mean()
high_missing_cols = null_pct_all[null_pct_all > 0.8].index.tolist()
df = df.drop(columns=high_missing_cols)

# Extract time features
if 'transaction_ts' in df.columns:
    ts = pd.to_datetime(df['transaction_ts'], errors='coerce')
    df['hour_of_day'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek
    df['is_night'] = (df['hour_of_day'].between(0, 5)).astype(int)

# Remove leakage columns
LEAKAGE_COLS = ['transaction_id', 'transaction_dt', 'transaction_ts']
existing_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
df = df.drop(columns=existing_leakage, errors='ignore')

# Separate X and y
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# Save feature list
features = X.columns.tolist()
joblib.dump(features, 'features.pkl')
print('features.pkl saved.')

# Detect col types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build preprocessor
num_pipeline = SklearnPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = SklearnPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ],
    remainder='drop'
)

print('Fitting preprocessor...')
X_train_processed = preprocessor.fit_transform(X_train)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')
print('preprocessor.pkl saved.')

# Prepare defaults
print('Computing defaults...')
defaults = {}
for col in num_cols:
    defaults[col] = float(X[col].median())
for col in cat_cols:
    defaults[col] = str(X[col].mode()[0])

joblib.dump(defaults, 'defaults.pkl')
print('defaults.pkl saved.')

# SMOTE
print('Applying SMOTE...')
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# Train Model
print('Training Random Forest...')
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_resampled, y_train_resampled)

joblib.dump(model, 'model_rf.pkl')
print('model_rf.pkl saved.')

print('Done! All pkl files have been recreated.')
