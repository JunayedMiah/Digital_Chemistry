import os
import warnings
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

warnings.filterwarnings('ignore')

# Config defaults   

SEED = 42
LMAX_MIN, LMAX_MAX = 200.0, 1100.0
TARGET_COL = "Absorption max (nm)"
SMILES_COL = "Chromophore"
SOLVENT_COL = "Solvent"


# Descriptor helpers

def smiles_to_mol(smi: str):
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required. Install via conda-forge.")
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None

def physchem_features_from_mol(mol) -> Dict[str, float]:
    
    if mol is None:
        return {
            "MolWt" : np.nan,
            "TPSA" : np.nan,
            "NumHDonors" : np.nan,
            "NumHAcceptors" : np.nan,
            "NumAromaticRings" : np.nan,
            "FractionCSP3" : np.nan,
            "MolLogP" : np.nan,
            "HeavyAtomCount" : np.nan,
            "NHOHCount" : np.nan,
            "NOCount" : np.nan,
        }
    return {
        "MolWt" : Descriptors.MolWt(mol),
        "TPSA" : Descriptors.TPSA(mol),
        "NumHDonors" : Descriptors.NumHDonors(mol),
        "NumHAcceptors" : Descriptors.NumHAcceptors(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "NumAliphaticRings" : Descriptors.NumAliphaticRings(mol),
        "FractionCSP3" : Descriptors.FractionCSP3(mol),
        "MolLogP" : Descriptors.MolLogP(mol),
        "HeavyAtomCount" : Descriptors.HeavyAtomCount(mol),
        "NHOHCount" : Descriptors.NHOHCount(mol),
    }
    
def morgan_fingerprint(mol, nBits: int = 2048, radius: int = 2) -> np.ndarray:
    
    arr = np.zeros(nBits, dtype=int)
    if mol is None: 
        return arr
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    ConvertToNumpyArray(bv, arr)
    return arr

def compute_descriptor_table(smiles_list: pd.Series) -> pd.DataFrame:
    
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required. Install via conda-forge.")
    
    mols = [smiles_to_mol(s) for s in smiles_list]
    
    # Physchem
    phys = [physchem_features_from_mol(m) for m in mols]
    phys_df = pd.DataFrame(phys)
    
    # Fingerprints
    fps = np.vstack([morgan_fingerprint(m) for m in mols])
    fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(fps.shape[1])])
    
    # Combine
    feats = pd.concat([phys_df, fp_df], axis=1)
    # Impute any NaNs in physchem with 0 (rare)
    feats = feats.fillna(0.0)
    return feats

# Data Loading and Cleaning

def load_and_clean(data_path: str) -> pd.DataFrame:
    df_raw = pd.read_csv(data_path)
    
    if TARGET_COL not in df_raw.columns or SMILES_COL not in df_raw.columns or SOLVENT_COL not in df_raw.columns:
        raise ValueError(f"CSV must include columns: {TARGET_COL}, {SMILES_COL}, {SOLVENT_COL}")
    
    df = df_raw.dropna(subset=[TARGET_COL, SMILES_COL]).copy()
    
    # Guard plausible wavelength range 
    df = df[(df[TARGET_COL] >= LMAX_MIN) & (df[TARGET_COL] <= LMAX_MAX)].copy()
    
    # Normalize types/whitespace
    df[SMILES_COL] = df[SMILES_COL].astype(str).str.strip()
    df[SOLVENT_COL] = df[SOLVENT_COL].astype(str).str.strip()
    df = df[df[SMILES_COL].str.len() > 0].copy()
    
    return df

# Feature building

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, OneHotEncoder]:
    
    # Chemistry descriptors
    Xchem = compute_descriptor_table(df[SMILES_COL].tolist())
    
    # One-Hot solvent
    ohe = OneHotEncoder(handle_unknown= "ignore", sparse_output= False)
    Xsol = ohe.fit_transform(df[[SOLVENT_COL]])
    Xsol = pd.DataFrame(
        Xsol,
        columns = [f"solvent_{c}" for c in ohe.get_feature_names_out([SOLVENT_COL])],
        index = df.index,
    )
    
    # Combine 
    X = pd.concat([Xchem.reset_index(drop=True), Xsol.reset_index(drop=True)], axis= 1)
    y = df[TARGET_COL].values.astype(float)
    groups = df[SMILES_COL].values.copy()  # Group by SMILES
    
    return X, y, groups, ohe


# CV Evaluation

def eval_cv(model, X: pd.DataFrame, y: np.ndarray, groups : np.ndarray, name: str = "model",
            n_splits: int = 5, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    gkf = GroupKFold(n_splits= n_splits)
    rmses, maes, r2s = [], [], []
    y_true_all, y_pred_all = [] , []
    
    for i, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        
        scaler = StandardScaler(with_mean= False)  # safer for sparse-like matrices
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        
        model.fit(Xtr_s, ytr)
        ypred = model.predict(Xte_s)
        
        rmse = mean_squared_error(yte, ypred, squared=False)
        mae = mean_absolute_error(yte, ypred)
        r2 = r2_score(yte, ypred)
        
        rmses.append(rmse); maes.append(mae); r2s.append(r2)
        y_true_all.extend(yte.tolist()); y_pred_all.extend(ypred.tolist())
        
        print(f"[{name}] Fold {i}: RMSE = {rmse:.2f} MAE = {mae:.2f} R2 = {r2:.3f}")
        
    metrics = {"RMSE" : float(np.mean(rmses)), "MAE": float(np.mean(maes)), "R2": float(np.mean(r2s))}
    print(f"[{name}] Mean : RMSE= {metrics['RMSE']:.2f} MAE = {metrics['MAE']:.2f} R2= {metrics['R2']:.3f}")
    return np.array(y_true_all) , np.array(y_pred_all), metrics


# Plot helpers 

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, title: str, outdir: str):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims)
    plt.xlabel("True λmax (nm)"); plt.ylabel("Predicted λmax (nm)"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title.replace(' ', '_').lower()}_pred_vs_true.png"), dpi=150)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str, outdir: str):
    plt.figure()
    resid = y_pred - y_true
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (Pred - True) (nm)"); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{title.replace(' ', '_').lower()}_residuals.png"), dpi=150)
    plt.close()

# Save and Inference

def train_final_and_save(best_model_name: str, model, X: pd.DataFrame, y: np.ndarray, ohe: OneHotEncoder, outdir: str, data_dir: str) -> str:
    
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)
    
    bundle = {
        "scaler" : scaler,
        "model" : model,
        "ohe_solvent_categories" : list(ohe.categories_[0]),
        "feature_columns" : list(X.columns),
        "info" : {"best_model": best_model_name},
    }
    os.makedirs(outdir, exist_ok = True)
    out_path = os.path.join(outdir, f"absorption_model_{best_model_name}.joblib")
    joblib.dump(bundle, out_path)
    return out_path

def predict_lmax(smiles: str, solvent: str, bundle: Dict[str, Any]) -> float:
    
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit required for inference.")
    scaler = bundle["scaler"]
    model = bundle["model"]
    feat_cols = bundle['feature_columns']
    sol_cats = bundle["ohe_solvent_categories"]
    
    mol = Chem.MolFromSmiles(smiles)
    feats = physchem_features_from_mol(mol)
    fp = morgan_fingerprint(mol)
    
    base = pd.DataFrame([feats])
    fp_df = pd.DataFrame([fp], columns=[c for c in feat_cols if c.startswith("fp_")])
    
    sol_vec = np.zeros((1, len(sol_cats)))
    if solvent in sol_cats:
        sol_vec[0, sol_cats.index(solvent)] = 1.0
    sol_df = pd.DataFrame(sol_vec, columns=[c for c in feat_cols if c.startswith("solvent_")])
    
    row = pd.concat([base.reset_index(drop=True), fp_df,], axis=1)
    row = row.reindex(columns=feat_cols, fill_value=0.0)
    
    Xs = scaler.transform(row)
    pred = model.predict(Xs)[0]
    return float(pred)

# main function

def main():
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not available")
    
    data_path = r"./Data/chromophore_data.csv"
    outdir = "./outputs"
    os.makedirs(outdir, exist_ok=True)
    
    print("Loading Data.......")
    df = load_and_clean(data_path)
    print("Cleanded shape:", df.shape)
    
    print("Building features.....")
    X, y, groups, ohe = build_features(df)
    print("Feature matrix:", X.shape)
    
    print("\n === Cross-validated evaluation ===")
    models = [
        ("Dummy", DummyRegressor(strategy="mean")),
        ("Ridge", Ridge(alpha=10.0, random_state=SEED)),
        ("RandomForest", RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)),
    ]
    results = {}
    for name, mdl in models:
        yt, yp, metrics = eval_cv(mdl, X, y, groups, name=name, n_splits=5)
        results[name] = {"y_true": yt, "y_pred": yp, "metrics": metrics}
        
        # plots
        
        plot_pred_vs_true(yt, yp, f"{name}", outdir)
        plot_residuals(yt, yp, f"{name}", outdir)
        
    # pick best by RMSE on stacked predictions
    best_name = None
    best_rmse = float("inf")
    for name, r in results.items():
        yt, yp = r["y_true"], r["y_pred"]
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            
    print(f"\nBest model by overall RMSE: {best_name} (RMSE = {best_rmse:.2f})")
    
    # Train-final and save
    best_model = [m for n, m in models if n == best_name][0]
    bundle_path = train_final_and_save(best_name, best_model, X, y, ohe, outdir, os.path.dirname(data_path))
    print("Saved model bundle to:", bundle_path)

    # Demo inference on first row
    try:
        example_smiles = df[SMILES_COL].iloc[0]
        example_solvent = df[SOLVENT_COL].iloc[0]
        bundle = joblib.load(bundle_path)
        pred_nm = predict_lmax(example_smiles, example_solvent, bundle)
        print(f"Example prediction for first row ({example_solvent}): {pred_nm:.1f} nm")
    except Exception as e:
        print("Inference demo failed:", e)


if __name__ == "__main__":
    main()
