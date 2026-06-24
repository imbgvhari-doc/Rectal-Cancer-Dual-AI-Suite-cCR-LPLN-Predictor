import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("Starting Unified Oncology AI Training Pipeline...")

# Load the REAL V5 Institutional Dataset
try:
    df = pd.read_excel("Anonimysed Rectal Cancer Data v5.xlsx")
    print(f"Successfully loaded institutional Excel dataset with {len(df)} records.\n")
except FileNotFoundError:
    try:
        df = pd.read_csv("Anonimysed Rectal Cancer Data v5.xlsx - Sheet3 (2).csv")
        print(f"Successfully loaded institutional CSV dataset with {len(df)} records.\n")
    except FileNotFoundError:
        print("Error: The V5 dataset (Excel or CSV) was not found in this folder. Please ensure it is in the Rectal_Cancer_AI folder.")
        exit()

# ====================================================
# MODEL 1: Complete Clinical Response (cCR) Predictor
# ====================================================
print("--- Training cCR (Watch & Wait) Model ---")
col_surgery = [c for c in df.columns if 'Surgery done' in c][0]
df['Target_cCR'] = df[col_surgery].astype(str).str.contains('Watch and Wait', case=False, na=False).astype(int)
y_ccr = df['Target_cCR']

col_age = [c for c in df.columns if 'Age' in c][0]
col_height = [c for c in df.columns if 'Verge' in c][0]
col_thick = [c for c in df.columns if 'Maximum wall' in c][0]

features_ccr = ['Age', 'Tumor_Height', 'Wall_Thickness']
df['Age'] = pd.to_numeric(df[col_age], errors='coerce')
df['Tumor_Height'] = pd.to_numeric(df[col_height], errors='coerce')
df['Wall_Thickness'] = pd.to_numeric(df[col_thick], errors='coerce')

X_ccr = df[features_ccr]
imputer_ccr = SimpleImputer(strategy='median')
X_ccr_imputed = pd.DataFrame(imputer_ccr.fit_transform(X_ccr), columns=features_ccr)

model_ccr = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, scale_pos_weight=(len(y_ccr)-sum(y_ccr))/sum(y_ccr), random_state=42)
model_ccr.fit(X_ccr_imputed, y_ccr)
joblib.dump(model_ccr, 'ccr_xgb_model.pkl')
joblib.dump(imputer_ccr, 'ccr_imputer.pkl')
print(f"cCR Model saved! Total cCR events: {sum(y_ccr)}\n")

# ====================================================
# MODEL 2: Lateral Pelvic Node (LPLN) Predictor (UPDATED WITH PET CT)
# ====================================================
print("--- Training LPLN Sterilization Model (PET Enhanced) ---")
col_node_pos = [c for c in df.columns if 'Extra mesorectal lymphnodes positivity' in c and 'initial MRI' in c][0]
df_lpln = df[df[col_node_pos].astype(str).str.strip().str.title() == 'Yes'].copy()

col_node_ctrt = [c for c in df.columns if 'after CTRT' in c][0]
def define_sterilization(val):
    val_str = str(val).lower()
    return 1 if 'no activity' in val_str or 'complete resolution' in val_str else 0
df_lpln['Target_LPLN'] = df_lpln[col_node_ctrt].apply(define_sterilization)
y_lpln = df_lpln['Target_LPLN']

# Extract MRI Size AND PET Positivity
col_node_size = [c for c in df.columns if 'Maximum Dimensions' in c and 'Lateral' in c][0]
col_pet_pos = [c for c in df.columns if 'Lateral pelvic' in c and 'PET' in c][-1] # Grabs the PET LPLN positivity column

df_lpln['Initial_LPLN_Size'] = pd.to_numeric(df_lpln[col_node_size], errors='coerce')
# Convert "Yes" on PET to 1, everything else to 0
df_lpln['PET_LPLN_Positive'] = df_lpln[col_pet_pos].astype(str).str.strip().str.title().apply(lambda x: 1 if 'Yes' in x else 0)

features_lpln = ['Age', 'Tumor_Height', 'Wall_Thickness', 'Initial_LPLN_Size', 'PET_LPLN_Positive']
X_lpln = df_lpln[features_lpln]

imputer_lpln = SimpleImputer(strategy='median')
X_lpln_imputed = pd.DataFrame(imputer_lpln.fit_transform(X_lpln), columns=features_lpln)

model_lpln = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42)
model_lpln.fit(X_lpln_imputed, y_lpln)
joblib.dump(model_lpln, 'lpln_xgb_model.pkl')
joblib.dump(imputer_lpln, 'lpln_imputer.pkl')
print(f"LPLN Model saved! Total Sterilization events: {sum(y_lpln)} out of {len(df_lpln)}\n")
print("Both models successfully trained and exported!")