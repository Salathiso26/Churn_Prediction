"""Bank churn project: Sections 1 to 8 in one script.

Put this file next to Churn_Modelling.csv and run one command at a time:

    python churn_app.py eda              Section 2: load, check and explore the data
    python churn_app.py baseline         Section 3: logistic vs random forest vs LightGBM
    python churn_app.py train            Section 3: train, evaluate and save the model
    python churn_app.py simulate-drift   Section 4: write a drifted batch (--severity 0 to 1)
    python churn_app.py check-drift      Section 4: PSI and KS report for a batch
    python churn_app.py auto-retrain     Section 4: retrain only if drift crosses the threshold
    python churn_app.py explain          Section 5: SHAP drivers for one example customer
    python churn_app.py api              Section 6: FastAPI server on port 8000
    python churn_app.py dashboard        Section 7: Streamlit dashboard on port 8501
    python churn_app.py test             Section 8: smoke tests
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)



# Section 1: Setup and configurations
BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(BASE, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS, "metrics.json")
REFERENCE_PATH = os.path.join(ARTIFACTS, "reference_sample.csv")
EDA_FIGURE = os.path.join(ARTIFACTS, "eda_overview.png")
BATCH_PATH = os.path.join(BASE, "current_batch.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

TARGET = "Exited"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY = ["HasCrCard", "IsActiveMember"]
CATEGORICAL = ["Geography", "Gender"]
FEATURE_COLUMNS = NUMERIC + BINARY + CATEGORICAL

ENG_NUMERIC = ["BalanceSalaryRatio", "TenureByAge", "ProductsPerTenure"]
ENG_BINARY = ["ZeroBalance"]

PSI_RETRAIN = 0.25  # above this: major shift, retrain
PSI_WARN = 0.10  # above this: moderate shift, watch

MLFLOW_EXPERIMENT = "bank-churn"
MLFLOW_MODEL_NAME = "bank-churn-classifier"

EXAMPLE_CUSTOMER = {
    "CreditScore": 608, "Geography": "Germany", "Gender": "Female", "Age": 58,
    "Tenure": 1, "Balance": 120000.0, "NumOfProducts": 3, "HasCrCard": 1,
    "IsActiveMember": 0, "EstimatedSalary": 90000.0,
}

# Section 2: Data preprocessing and engineering


def load_and_check(path="Churn_Modelling.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Class counts:\n", df["Exited"].value_counts())
    return df


def add_engineered_features(df):
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1.0)
    df["TenureByAge"] = df["Tenure"] / (df["Age"] - 17.0).clip(lower=1)
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1.0)
    return df


def prepare_data(df):
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = add_engineered_features(df)
    return df


def split_data(df):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, stratify=y_trainval, random_state=RANDOM_STATE
    )
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test



# Section 3: Model building, evaluation and training

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC + ENG_NUMERIC),
            ("bin", "passthrough", BINARY + ENG_BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(params=None):
    settings = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        settings.update(params)
    return Pipeline([("preprocess", build_preprocessor()), ("clf", LGBMClassifier(**settings))])


def evaluate_model(y_true, y_proba, threshold=0.5):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": threshold,
    }


def log_to_mlflow(model, val_metrics, test_metrics):
    # optional: if mlflow is missing or fails, training still succeeds
    try:
        import mlflow
        import mlflow.sklearn

        db_path = os.path.join(ARTIFACTS, "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri("sqlite:///" + db_path)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run() as run:
            mlflow.log_params(model.named_steps["clf"].get_params())
            mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
            mlflow.log_metrics({"test_" + k: v for k, v in test_metrics.items()})
            mlflow.log_artifact(METRICS_PATH)
            mlflow.sklearn.log_model(model, name="model", registered_model_name=MLFLOW_MODEL_NAME)
            print("MLflow run logged:", run.info.run_id)
            print("Browse with: mlflow ui --backend-store-uri sqlite:///" + db_path)
    except ImportError:
        print("MLflow not installed, skipping experiment tracking (pip install mlflow)")
    except Exception as e:
        print("MLflow skipped:", type(e).__name__, e)


def train_model(params=None):
    df = prepare_data(load_and_check())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    model = build_model(params)
    model.fit(X_train, y_train)

    val_metrics = evaluate_model(y_val, model.predict_proba(X_val)[:, 1])
    test_metrics = evaluate_model(y_test, model.predict_proba(X_test)[:, 1])

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "params": params or {}}, f, indent=2)
    sample_size = min(2000, len(X_train))
    X_train.sample(sample_size, random_state=RANDOM_STATE).to_csv(REFERENCE_PATH, index=False)

    print("Test ROC-AUC:", round(test_metrics["roc_auc"], 4))
    print("Test PR-AUC:", round(test_metrics["pr_auc"], 4))
    print("Test recall:", round(test_metrics["recall"], 4))
    log_to_mlflow(model, val_metrics, test_metrics)
    return {"val": val_metrics, "test": test_metrics}


def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model found. Run: python churn_app.py train")
    return joblib.load(MODEL_PATH)



# Section 4: Data drift detection (PSI and KS test)

def psi_for_column(expected, actual, bins=10):
    expected = expected[~pd.isnull(expected)]
    actual = actual[~pd.isnull(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_status(psi):
    if psi >= PSI_RETRAIN:
        return "MAJOR_SHIFT"
    if psi >= PSI_WARN:
        return "MODERATE_SHIFT"
    return "STABLE"


def compute_drift(reference, current):
    rows = []
    for col in FEATURE_COLUMNS:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col]
        cur = current[col]
        if not pd.api.types.is_numeric_dtype(ref):
            cats = ref.astype("category").cat.categories
            ref_values = ref.astype("category").cat.codes.to_numpy()
            cur_values = pd.Categorical(cur, categories=cats).codes
            ks_p = np.nan
        else:
            ref_values = ref.to_numpy(dtype=float)
            cur_values = cur.to_numpy(dtype=float)
            ks_p = float(ks_2samp(ref_values, cur_values).pvalue) if col in NUMERIC else np.nan
        psi = psi_for_column(ref_values, cur_values)
        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "ks_pvalue": None if np.isnan(ks_p) else round(ks_p, 4),
            "status": psi_status(psi),
        })
    report = pd.DataFrame(rows)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)


def should_retrain(report):
    if len(report) == 0:
        return False, "no features compared"
    worst = report.iloc[0]
    if worst["psi"] >= PSI_RETRAIN:
        return True, f"{worst['feature']} PSI={worst['psi']} >= {PSI_RETRAIN}"
    return False, f"max PSI {worst['psi']} ({worst['feature']}) is below {PSI_RETRAIN}"


def simulate_drift(severity=1.0):
    # the public dataset is static, so inject a shift: older, richer, weaker-credit customers
    df = load_and_check().copy()
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(df)
    df["Age"] = (df["Age"] + rng.normal(8 * severity, 2, n)).clip(18, 95).round().astype(int)
    df["Balance"] = (df["Balance"] * (1 + 0.4 * severity)).clip(0, 400000)
    df["CreditScore"] = (df["CreditScore"] - rng.normal(30 * severity, 5, n)).clip(350, 850).round().astype(int)
    flip = rng.random(n) < 0.15 * severity
    df.loc[flip, "NumOfProducts"] = df.loc[flip, "NumOfProducts"].clip(upper=3) + 1
    df.to_csv(BATCH_PATH, index=False)
    print("Wrote", BATCH_PATH, "with severity", severity)
    return BATCH_PATH


def drift_report_for_batch(batch_path):
    if not os.path.exists(REFERENCE_PATH):
        sys.exit("No reference sample found. Run: python churn_app.py train")
    if not os.path.exists(batch_path):
        sys.exit(batch_path + " not found. Run: python churn_app.py simulate-drift")
    reference = pd.read_csv(REFERENCE_PATH)
    current = prepare_data(pd.read_csv(batch_path)).drop(columns=[TARGET], errors="ignore")
    report = compute_drift(reference, current)
    retrain, reason = should_retrain(report)
    return report, retrain, reason


# Section 5: Explainability (SHAP)

def make_explainer(model):
    import shap

    return shap.TreeExplainer(model.named_steps["clf"])


def explain_row(model, explainer, row):
    # positive SHAP value pushes the customer toward churn, negative pushes away
    pre = model.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())
    Xt = pre.transform(row)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = sv.reshape(-1)[: len(feature_names)]
    out = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "value": np.asarray(Xt).reshape(-1),
    })
    out["abs"] = out["shap_value"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs")
    return out.reset_index(drop=True)


def explainer_base_value(explainer):
    return float(np.asarray(explainer.expected_value).reshape(-1)[-1])


def risk_band(p):
    if p >= 0.5:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"



# Section 6: Serving layer (FastAPI)

def new_fastapi(**kwargs):
    # Newer Streamlit scans a script for "x = FastAPI(...)" and then tries to launch the
    # whole file as an API server. Creating it here keeps the dashboard launchable.
    from fastapi import FastAPI

    return FastAPI(**kwargs)


def build_api():
    from fastapi import HTTPException
    from pydantic import BaseModel, Field

    model = load_model()
    explainer = make_explainer(model)
    api = new_fastapi(title="Bank Churn API", version="1.0.0")

    class Customer(BaseModel):
        CreditScore: int = Field(..., ge=300, le=900, examples=[619])
        Geography: str = Field(..., examples=["France"])
        Gender: str = Field(..., examples=["Female"])
        Age: int = Field(..., ge=18, le=120, examples=[42])
        Tenure: int = Field(..., ge=0, le=10, examples=[2])
        Balance: float = Field(..., ge=0, examples=[0.0])
        NumOfProducts: int = Field(..., ge=1, le=4, examples=[1])
        HasCrCard: int = Field(..., ge=0, le=1, examples=[1])
        IsActiveMember: int = Field(..., ge=0, le=1, examples=[1])
        EstimatedSalary: float = Field(..., ge=0, examples=[101348.88])

    class Batch(BaseModel):
        records: list[Customer]

    def to_frame(customer):
        return add_engineered_features(pd.DataFrame([customer.model_dump()]))

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "n_features": len(FEATURE_COLUMNS)}

    @api.post("/predict")
    def predict(customer: Customer):
        p = float(model.predict_proba(to_frame(customer))[0, 1])
        return {"churn_probability": round(p, 4), "risk_band": risk_band(p)}

    @api.post("/predict/explain")
    def predict_explain(customer: Customer, top_n: int = 6):
        X = to_frame(customer)
        p = float(model.predict_proba(X)[0, 1])
        top = explain_row(model, explainer, X).head(top_n)
        drivers = []
        for r in top.itertuples():
            drivers.append({
                "feature": r.feature,
                "shap_value": round(float(r.shap_value), 4),
                "direction": "increases_risk" if r.shap_value > 0 else "decreases_risk",
            })
        return {
            "churn_probability": round(p, 4),
            "risk_band": risk_band(p),
            "base_value": round(explainer_base_value(explainer), 4),
            "top_drivers": drivers,
        }

    @api.post("/drift")
    def drift(batch: Batch):
        if not os.path.exists(REFERENCE_PATH):
            raise HTTPException(503, "No reference distribution. Train the model first.")
        reference = pd.read_csv(REFERENCE_PATH)
        current = add_engineered_features(pd.DataFrame([r.model_dump() for r in batch.records]))
        report = compute_drift(reference, current)
        retrain, reason = should_retrain(report)
        # JSON cannot hold NaN (the KS p-value is NaN for categories), so send null instead
        clean = report.astype(object).where(report.notna(), None)
        return {
            "retrain_recommended": retrain,
            "reason": reason,
            "report": clean.to_dict(orient="records"),
        }

    return api



# Section 7: Dashboard (Streamlit)

def dashboard_ui():
    import streamlit as st

    st.set_page_config(page_title="Bank Churn Monitor", page_icon="📉", layout="wide")

    @st.cache_resource
    def load_everything():
        m = load_model()
        return m, make_explainer(m)

    def show_table(data, **kwargs):
        # works on old and new Streamlit versions
        try:
            st.dataframe(data, width="stretch", **kwargs)
        except TypeError:
            st.dataframe(data, use_container_width=True, **kwargs)

    model, explainer = load_everything()

    st.title("Bank Churn: Risk and Explainability")
    tab_single, tab_batch = st.tabs(["Single customer", "Portfolio and drift"])

    with tab_single:
        c1, c2, c3 = st.columns(3)
        with c1:
            credit = st.slider("Credit score", 350, 850, 619)
            age = st.slider("Age", 18, 92, 42)
            tenure = st.slider("Tenure (yrs)", 0, 10, 2)
        with c2:
            balance = st.number_input("Balance", 0.0, 300000.0, 0.0, step=1000.0)
            salary = st.number_input("Estimated salary", 0.0, 200000.0, 101348.0, step=1000.0)
            products = st.selectbox("Num products", [1, 2, 3, 4], index=0)
        with c3:
            geo = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            has_card = st.checkbox("Has credit card", True)
            active = st.checkbox("Active member", True)

        row = pd.DataFrame([{
            "CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
            "HasCrCard": int(has_card), "IsActiveMember": int(active),
            "EstimatedSalary": salary,
        }])
        X = add_engineered_features(row)
        proba = float(model.predict_proba(X)[0, 1])
        band = risk_band(proba)
        icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[band]
        st.metric("Churn probability", f"{proba:.1%}", help="P(customer leaves)")
        st.subheader(f"{icon} {band} risk")

        st.markdown("#### Why: top drivers (SHAP)")
        top = explain_row(model, explainer, X).head(8)
        top["effect"] = top["shap_value"].apply(lambda v: "increases risk" if v > 0 else "lowers risk")
        show_table(
            top[["feature", "shap_value", "effect"]],
            hide_index=True,
            column_config={"shap_value": st.column_config.NumberColumn(format="%.3f")},
        )
        st.bar_chart(top.set_index("feature")["shap_value"])

    with tab_batch:
        st.markdown(
            "Upload a CSV of customers (raw schema, target not required) to score the "
            "portfolio and check data drift against the training distribution."
        )
        up = st.file_uploader("Customer batch CSV", type=["csv"])
        if up is not None:
            raw = pd.read_csv(up)
            feats = add_engineered_features(raw)
            feats["churn_probability"] = model.predict_proba(feats)[:, 1]

            a, b = st.columns(2)
            a.metric("Customers", len(feats))
            a.metric("Predicted high-risk (50% or more)", int((feats["churn_probability"] >= 0.5).sum()))

            bucket_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
            buckets = pd.cut(
                feats["churn_probability"],
                bins=[0, 0.25, 0.5, 0.75, 1],
                labels=bucket_labels,
                include_lowest=True,
            )
            # text labels (not Interval objects) so the Streamlit chart can draw them
            b.bar_chart(buckets.value_counts().reindex(bucket_labels))

            if os.path.exists(REFERENCE_PATH):
                st.markdown("#### Data drift vs training reference")
                report = compute_drift(pd.read_csv(REFERENCE_PATH), feats)
                retrain, reason = should_retrain(report)
                if retrain:
                    st.error("Retraining recommended: " + reason)
                else:
                    st.success("Distribution stable: " + reason)
                show_table(report, hide_index=True)

            st.download_button("Download scored CSV", feats.to_csv(index=False),
                               "scored_customers.csv", "text/csv")



# Section 8: Commands, tests and entry point

def run_eda():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_and_check()
    print("Churn rate:", f"{df[TARGET].mean():.1%}")
    print("\nChurn rate by segment:")
    for col in ["Geography", "Gender", "NumOfProducts", "IsActiveMember"]:
        print(" ", col)
        for k, v in df.groupby(col)[TARGET].mean().round(3).items():
            print("   ", k, f"{v:.1%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.groupby("Geography")[TARGET].mean().plot.bar(ax=axes[0], title="Churn by geography")
    df.boxplot(column="Age", by=TARGET, ax=axes[1])
    axes[1].set_title("Age vs churn")
    axes[1].figure.suptitle("")
    df.groupby("NumOfProducts")[TARGET].mean().plot.bar(ax=axes[2], title="Churn by number of products")
    plt.tight_layout()
    plt.savefig(EDA_FIGURE, dpi=110)
    print("\nSaved", EDA_FIGURE)


def run_baseline():
    df = prepare_data(load_and_check())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    def wrap(classifier):
        return Pipeline([("preprocess", build_preprocessor()), ("clf", classifier)])

    models = {
        "logistic": wrap(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": wrap(RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
        "lightgbm": build_model(),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        metrics = evaluate_model(y_test, m.predict_proba(X_test)[:, 1])
        rows.append({"model": name, **{k: round(v, 4) for k, v in metrics.items()}})
    print(pd.DataFrame(rows).to_string(index=False))


def run_check_drift(batch):
    report, retrain, reason = drift_report_for_batch(batch or BATCH_PATH)
    print(report.to_string(index=False))
    print("\nRetrain recommended:", retrain, "(" + reason + ")")


def run_auto_retrain(batch):
    batch = batch or BATCH_PATH
    if not os.path.exists(MODEL_PATH) or not os.path.exists(REFERENCE_PATH):
        print("No model or reference yet, training from scratch")
        train_model()
        return
    report, retrain, reason = drift_report_for_batch(batch)
    print(report.to_string(index=False))
    print("\nDecision:", reason)
    if retrain:
        print("Drift above threshold, retraining")
        train_model()
    else:
        print("Drift below threshold, keeping the current model")


def run_explain():
    model = load_model()
    explainer = make_explainer(model)
    X = add_engineered_features(pd.DataFrame([EXAMPLE_CUSTOMER]))
    p = float(model.predict_proba(X)[0, 1])
    print("Example customer:", EXAMPLE_CUSTOMER)
    print(f"Churn probability: {p:.1%} ({risk_band(p)} risk)\n")
    print(explain_row(model, explainer, X).head(8).to_string(index=False))


def run_api(port):
    import uvicorn

    uvicorn.run(build_api(), host="0.0.0.0", port=port)


def run_dashboard(port):
    env = dict(os.environ, CHURN_MODE="dashboard")
    script = os.path.abspath(__file__)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(port)],
        env=env,
        check=False,
    )


def run_tests():
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:
            results.append((name, False, type(e).__name__ + ": " + str(e)))

    sample = prepare_data(load_and_check()).head(500)
    reference = sample.drop(columns=[TARGET])

    def test_features():
        for col in ["BalanceSalaryRatio", "TenureByAge", "ZeroBalance", "ProductsPerTenure"]:
            assert col in sample.columns, col

    def test_model():
        m = build_model({"n_estimators": 50})
        m.fit(reference, sample[TARGET])
        proba = m.predict_proba(reference)[:, 1]
        assert len(proba) == len(sample)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_no_drift():
        report = compute_drift(reference, reference)
        assert (report["psi"] < PSI_WARN).all()
        assert should_retrain(report)[0] is False

    def test_drift():
        shifted = reference.copy()
        shifted["Age"] = shifted["Age"] + 15
        assert should_retrain(compute_drift(reference, shifted))[0] is True

    def test_explain():
        m = build_model({"n_estimators": 50})
        m.fit(reference, sample[TARGET])
        explainer = make_explainer(m)
        row = add_engineered_features(pd.DataFrame([EXAMPLE_CUSTOMER]))
        out = explain_row(m, explainer, row)
        assert len(out) == len(m.named_steps["preprocess"].get_feature_names_out())
        assert out["shap_value"].notna().all()

    check("engineered features exist", test_features)
    check("model trains and predicts in [0, 1]", test_model)
    check("no drift on identical data", test_no_drift)
    check("drift detected on shifted data", test_drift)
    check("SHAP explanation shape", test_explain)

    print()
    for name, ok, msg in results:
        print("[PASS]" if ok else "[FAIL]", name, msg)
    failed = sum(1 for r in results if not r[1])
    print(f"\n{len(results) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

def main():
    parser = argparse.ArgumentParser(description="Bank churn project in one script")
    parser.add_argument("command", choices=[
        "eda", "baseline", "train", "simulate-drift", "check-drift",
        "auto-retrain", "explain", "api", "dashboard", "test",
    ])
    parser.add_argument("--severity", type=float, default=1.0, help="simulate-drift: 0 = none, 1 = strong")
    parser.add_argument("--batch", default=None, help="check-drift / auto-retrain: CSV path")
    parser.add_argument("--port", type=int, default=None, help="api (default 8000) / dashboard (default 8501)")
    args = parser.parse_args()

    if args.command == "eda":
        run_eda()
    elif args.command == "baseline":
        run_baseline()
    elif args.command == "train":
        train_model()
    elif args.command == "simulate-drift":
        simulate_drift(args.severity)
    elif args.command == "check-drift":
        run_check_drift(args.batch)
    elif args.command == "auto-retrain":
        run_auto_retrain(args.batch)
    elif args.command == "explain":
        run_explain()
    elif args.command == "api":
        run_api(args.port or 8000)
    elif args.command == "dashboard":
        run_dashboard(args.port or 8501)
    elif args.command == "test":
        run_tests()


# Streamlit runs this file top to bottom with __name__ == "__main__",
# so the dashboard flag has to be checked first.
if os.environ.get("CHURN_MODE") == "dashboard":
    dashboard_ui()
elif __name__ == "__main__":
    main()
