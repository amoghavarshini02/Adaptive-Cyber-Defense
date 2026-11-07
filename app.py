 
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Adaptive Cyber Defense | SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CLASS_NAMES = {
    0: "Benign",
    1: "Exfiltration",
    2: "Initial access",
    3: "Lateral movement",
    4: "Reconnaissance",
}


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080e1a;
    color: #e5edf7;
}

[data-testid="stSidebar"] {
    background: #0d1728;
    border-right: 1px solid #23344b;
}

[data-testid="stSidebar"] hr {
    border-color: #344256;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

.hero {
    background: linear-gradient(120deg, #12243c, #102d40);
    border: 1px solid #24445e;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #f8fafc;
    margin: 12px 0 8px;
}

.hero-subtitle {
    color: #a8bfd4;
    font-size: 14px;
    line-height: 1.7;
}

.status-pill {
    display: inline-block;
    background: #183b3b;
    color: #5eead4;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .7px;
    border: 1px solid #27605c;
}

.metric-card {
    background: #101c2d;
    border: 1px solid #23344b;
    border-radius: 14px;
    padding: 19px 20px;
    min-height: 120px;
}

.metric-label {
    color: #91a5bc;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 13px;
    text-transform: uppercase;
    letter-spacing: .6px;
}

.metric-value {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 800;
}

.metric-note {
    color: #8298b0;
    font-size: 11px;
    margin-top: 7px;
}

.section-label {
    color: #e2e8f0;
    font-size: 17px;
    font-weight: 700;
    margin-top: 27px;
    margin-bottom: 14px;
}

.info-panel {
    background: #101c2d;
    border: 1px solid #23344b;
    border-radius: 12px;
    padding: 20px;
    color: #cbd5e1;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #23344b;
    border-radius: 10px;
}

div[data-testid="stFileUploader"] {
    background: #101c2d;
    border: 1px solid #23344b;
    border-radius: 12px;
    padding: 15px;
}

footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load_json(filename):
    path = REPORTS / filename

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        st.error(f"Could not read {filename}: {error}")
        return None


@st.cache_data
def load_csv(filename):
    path = REPORTS / filename

    if not path.exists():
        return None

    try:
        if filename == "classification_metrics.csv":
            df = pd.read_csv(path, index_col=0)
            df.index.name = "Threat Class"
            return df

        df = pd.read_csv(path)
        df = df.loc[
            :,
            ~df.columns.astype(str).str.match(r"^Unnamed")
        ]
        return df

    except Exception as error:
        st.error(f"Could not read {filename}: {error}")
        return None


def normalize_key(value):
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "")
    )


def recursive_metric_search(data, aliases):
    """Search metric names inside nested dictionaries."""

    if not isinstance(data, dict):
        return None

    normalized_aliases = [normalize_key(x) for x in aliases]

    for key, value in data.items():
        if normalize_key(key) in normalized_aliases:
            if not isinstance(value, (dict, list)):
                return value

    for value in data.values():
        if isinstance(value, dict):
            result = recursive_metric_search(value, aliases)
            if result is not None:
                return result

    return None


def find_metric(data, names):
    return recursive_metric_search(data, names)


def find_group(data, group_names):
    """Find a nested baseline/prequential result group."""

    if not isinstance(data, dict):
        return None

    aliases = [normalize_key(x) for x in group_names]

    for key, value in data.items():
        if normalize_key(key) in aliases and isinstance(value, dict):
            return value

    for value in data.values():
        if isinstance(value, dict):
            result = find_group(value, group_names)
            if result is not None:
                return result

    return None


def find_group_metric(data, group_names, metric_names):
    """Find a metric inside a named result group."""

    group = find_group(data, group_names)

    if group is not None:
        result = find_metric(group, metric_names)
        if result is not None:
            return result

    # Fallback for flat JSON structures.
    return find_metric(data, metric_names)


def format_percent(value):
    if value is None:
        return "—"

    try:
        value = float(value)

        if abs(value) <= 1:
            value *= 100

        return f"{value:.2f}%"

    except (TypeError, ValueError):
        return "—"


# Load experiment results.

metrics = load_json("metrics.json")
prequential = load_json("prequential_results.json")
drift = load_json("drift_detection_results.json")

batch_results = load_csv("prequential_batch_results.csv")
confusion = load_csv("confusion_matrix.csv")
class_metrics = load_csv("classification_metrics.csv")


# =========================================================
# 4. REUSABLE UI COMPONENTS
# =========================================================

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f'<div class="section-label">{title}</div>',
        unsafe_allow_html=True,
    )


def plot_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="#101c2d",
        plot_bgcolor="#101c2d",
        font=dict(color="#cbd5e1", family="Inter"),
        margin=dict(l=25, r=25, t=55, b=30),
    )

    fig.update_xaxes(gridcolor="#24344a")
    fig.update_yaxes(gridcolor="#24344a")

    return fig


# =========================================================
# 5. CLASSIFICATION TABLE
# =========================================================

def clean_classification_table(df):
    if df is None or df.empty:
        return None

    df = df.copy()
    df.index.name = "Threat Class"
    df = df.reset_index()

    df = df.loc[
        :,
        ~df.columns.astype(str).str.match(r"^Unnamed")
    ]

    for col in ["precision", "recall", "f1-score"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce") * 100
            ).round(2)

    df = df.rename(columns={
        "precision": "Precision (%)",
        "recall": "Recall (%)",
        "f1-score": "F1 Score (%)",
        "support": "Support",
    })

    return df


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🛡️ CYBER DEFENSE")
    st.caption("Adaptive AI Security Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Security Overview",
            "Model Performance",
            "Online Learning",
            "Drift Detection",
            "Prediction Lab",
        ],
    )

    st.divider()

    st.markdown("### PROJECT STATUS")
    st.markdown("Dataset: MedSec-25")
    st.markdown("Model: SGD Classifier")
    st.markdown("Learning: Incremental")
    st.markdown("Drift: ADWIN")

    st.divider()

    st.caption("Offline experiment dashboard")
    st.caption("Saved evaluation artifacts")


# =========================================================
# 7. DASHBOARD HEADER
# =========================================================

st.html("""
<div class="hero">

    <div class="status-pill">
        OFFLINE EXPERIMENT • EVALUATION MODE
    </div>

    <div class="hero-title">
        Adaptive AI-Based Cyber Defense
    </div>

    <div class="hero-subtitle">
        Intelligent network threat classification, incremental learning
        and potential concept-drift monitoring for IoMT environments.
    </div>

</div>
""")


# =========================================================
# PAGE 1: SECURITY OVERVIEW
# =========================================================

if page == "Security Overview":

    section("System Performance")

    accuracy = find_metric(metrics, ["accuracy", "test_accuracy"])
    balanced = find_metric(metrics, ["balanced_accuracy"])
    macro_f1 = find_metric(metrics, ["macro_f1", "f1_macro"])

    drift_count = find_metric(
        drift,
        [
            "drift_events_detected",
            "drift_events",
            "number_of_drifts",
            "detected_events",
        ],
    )

    if drift_count is None and isinstance(drift, dict):
        drift_count = len(drift.get("events", []))

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Baseline Accuracy", format_percent(accuracy))

    with c2:
        metric_card("Balanced Accuracy", format_percent(balanced))

    with c3:
        metric_card("Macro F1 Score", format_percent(macro_f1))

    with c4:
        metric_card(
            "Potential Drift Events",
            str(drift_count) if drift_count is not None else "—",
            "ADWIN detection output",
        )

    section("Threat Classification")

    clean_table = clean_classification_table(class_metrics)

    if clean_table is not None:
        st.dataframe(
            clean_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precision (%)": st.column_config.NumberColumn(
                    "Precision (%)", format="%.2f%%"
                ),
                "Recall (%)": st.column_config.NumberColumn(
                    "Recall (%)", format="%.2f%%"
                ),
                "F1 Score (%)": st.column_config.NumberColumn(
                    "F1 Score (%)", format="%.2f%%"
                ),
                "Support": st.column_config.NumberColumn(
                    "Support", format="%d"
                ),
            },
        )
    else:
        st.warning("Classification metrics file is unavailable.")

    section("System Architecture")

    st.html("""
    <div class="info-panel">
        <div style="
            display:flex;
            justify-content:space-around;
            align-items:center;
            flex-wrap:wrap;
            gap:16px;
            text-align:center;
        ">
            <div>
                <b style="color:#38bdf8;">MedSec-25</b><br>
                <small>IoMT Dataset</small>
            </div>
            <div style="color:#64748b;">→</div>
            <div>
                <b style="color:#38bdf8;">Preprocessing</b><br>
                <small>Feature preparation</small>
            </div>
            <div style="color:#64748b;">→</div>
            <div>
                <b style="color:#38bdf8;">ML Classifier</b><br>
                <small>Threat classification</small>
            </div>
            <div style="color:#64748b;">→</div>
            <div>
                <b style="color:#38bdf8;">Online Learning</b><br>
                <small>Incremental updates</small>
            </div>
            <div style="color:#64748b;">→</div>
            <div>
                <b style="color:#facc15;">ADWIN</b><br>
                <small>Drift monitoring</small>
            </div>
        </div>
    </div>
    """)


# =========================================================
# PAGE 2: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    section("Baseline Model Evaluation")

    accuracy = find_metric(metrics, ["accuracy", "test_accuracy"])
    balanced = find_metric(metrics, ["balanced_accuracy"])
    macro_f1 = find_metric(metrics, ["macro_f1", "f1_macro"])

    a, b, c = st.columns(3)

    with a:
        metric_card("Accuracy", format_percent(accuracy))

    with b:
        metric_card("Balanced Accuracy", format_percent(balanced))

    with c:
        metric_card("Macro F1", format_percent(macro_f1))

    section("Class-wise Performance")

    clean_table = clean_classification_table(class_metrics)

    if clean_table is not None:
        st.dataframe(
            clean_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Classification metrics are unavailable.")

    section("Confusion Matrix")

    if confusion is not None and not confusion.empty:

        numeric = confusion.select_dtypes(include=np.number)

        if not numeric.empty:

            fig = px.imshow(
                numeric,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={
                    "x": "Predicted class",
                    "y": "Actual class",
                    "color": "Records",
                },
            )

            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(len(CLASS_NAMES))),
                ticktext=list(CLASS_NAMES.values()),
            )

            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(CLASS_NAMES))),
                ticktext=list(CLASS_NAMES.values()),
            )

            st.plotly_chart(
                plot_layout(fig, 450),
                use_container_width=True,
            )

        else:
            st.dataframe(confusion, use_container_width=True)

    else:
        st.info("Confusion matrix is unavailable.")


# =========================================================
# PAGE 3: ONLINE LEARNING
# =========================================================

elif page == "Online Learning":

    section("Baseline vs Online Learning")

    # Read metrics from either nested or flat JSON formats.

    baseline_accuracy = find_group_metric(
        prequential,
        ["baseline", "baseline_metrics", "baseline_results"],
        ["accuracy", "baseline_accuracy"],
    )

    online_accuracy = find_group_metric(
        prequential,
        ["prequential", "prequential_metrics",
         "online", "online_metrics", "online_results"],
        ["accuracy", "prequential_accuracy", "online_accuracy"],
    )

    baseline_balanced = find_group_metric(
        prequential,
        ["baseline", "baseline_metrics", "baseline_results"],
        ["balanced_accuracy", "baseline_balanced_accuracy"],
    )

    online_balanced = find_group_metric(
        prequential,
        ["prequential", "prequential_metrics",
         "online", "online_metrics", "online_results"],
        [
            "balanced_accuracy",
            "prequential_balanced_accuracy",
            "online_balanced_accuracy",
        ],
    )

    baseline_f1 = find_group_metric(
        prequential,
        ["baseline", "baseline_metrics", "baseline_results"],
        ["macro_f1", "f1_macro", "baseline_macro_f1"],
    )

    online_f1 = find_group_metric(
        prequential,
        ["prequential", "prequential_metrics",
         "online", "online_metrics", "online_results"],
        [
            "macro_f1",
            "f1_macro",
            "prequential_macro_f1",
            "online_macro_f1",
        ],
    )

    comparisons = [
        ("Accuracy", baseline_accuracy, online_accuracy),
        ("Balanced Accuracy", baseline_balanced, online_balanced),
        ("Macro F1", baseline_f1, online_f1),
    ]

    for label, before, after in comparisons:

        st.markdown(f"**{label}**")

        left, right = st.columns(2)

        with left:
            metric_card("Baseline", format_percent(before))

        with right:
            metric_card("Prequential", format_percent(after))

    # If the file format still cannot be interpreted, show a clear warning.
    if any(
        value is None
        for value in [
            baseline_accuracy,
            online_accuracy,
            baseline_balanced,
            online_balanced,
            baseline_f1,
            online_f1,
        ]
    ):
        st.warning(
            "Some comparison metrics could not be read from "
            "prequential_results.json. Check the JSON structure."
        )

    section("Batch-wise Learning Behaviour")

    if batch_results is not None and not batch_results.empty:

        st.caption(
            "Metrics recorded during the test-then-train batch simulation."
        )

        batch_column = next(
            (
                col for col in batch_results.columns
                if col.lower() == "batch"
            ),
            batch_results.columns[0],
        )

        metric_columns = [
            col for col in batch_results.columns
            if any(
                term in col.lower()
                for term in ["accuracy", "macro_f1", "f1"]
            )
        ]

        if metric_columns:

            fig = go.Figure()

            for col in metric_columns:

                fig.add_trace(
                    go.Scatter(
                        x=batch_results[batch_column],
                        y=batch_results[col],
                        mode="lines+markers",
                        name=col.replace("_", " ").title(),
                    )
                )

            fig.update_layout(
                title="Performance Across Learning Batches",
                xaxis_title="Batch",
                yaxis_title="Score",
            )

            st.plotly_chart(
                plot_layout(fig, 400),
                use_container_width=True,
            )

        st.dataframe(
            batch_results,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Prequential batch results are unavailable.")

    st.info(
        "This experiment uses a labelled holdout stream for "
        "test-then-train simulation. It is not a live network feed."
    )


# =========================================================
# PAGE 4: DRIFT DETECTION
# =========================================================

elif page == "Drift Detection":

    section("ADWIN Concept-Drift Monitoring")

    if drift is not None:

        count = find_metric(
            drift,
            [
                "drift_events_detected",
                "drift_events",
                "number_of_drifts",
                "detected_events",
            ],
        )

        events = drift.get("events", [])

        if count is None:
            count = len(events) if isinstance(events, list) else 0

        records = find_metric(
            drift,
            ["records_monitored", "total_records_monitored"],
        )

        left, right = st.columns(2)

        with left:
            metric_card(
                "Potential Drift Events",
                str(count),
                "Reported by ADWIN",
            )

        with right:
            metric_card(
                "Records Monitored",
                f"{int(records):,}" if records is not None else "—",
                "Saved experiment output",
            )

        section("Detection Event Details")

        if events:
            event_df = pd.DataFrame(events)

            st.dataframe(
                event_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No event details are available.")

        with st.expander("View complete detector configuration"):
            st.json(drift)

    else:
        st.warning("drift_detection_results.json was not found.")

    st.html("""
    <div class="info-panel">
        <b style="color:#facc15;">Interpretation</b>
        <p>
        ADWIN monitors changes in the observed prediction-error stream.
        A detected event is a potential drift signal. It does not,
        by itself, prove real-world network traffic drift.
        </p>
        <p>
        The current implementation records drift events. It does not
        automatically reset or retrain the model after detection.
        </p>
    </div>
    """)


# =========================================================
# PAGE 5: PREDICTION LAB
# =========================================================

elif page == "Prediction Lab":

    section("Single-record Threat Classification")

    st.markdown("""
    Upload a CSV containing the model's exact feature columns.
    The saved baseline model will classify the supplied records.
    This is offline file-based inference, not live packet capture.
    """)

    uploaded_file = st.file_uploader(
        "Upload feature CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            input_df = pd.read_csv(uploaded_file)

            model_path = MODELS / "cyber_model.pkl"
            scaler_path = MODELS / "scaler.pkl"
            features_path = MODELS / "feature_columns.pkl"

            required_paths = [
                model_path,
                scaler_path,
                features_path,
            ]

            if not all(path.exists() for path in required_paths):

                st.error("Required saved model files are missing.")

            else:

                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                feature_columns = joblib.load(features_path)

                missing = [
                    col for col in feature_columns
                    if col not in input_df.columns
                ]

                if missing:

                    st.error(
                        f"Missing {len(missing)} required feature columns."
                    )

                    st.code(", ".join(missing))

                else:

                    X = input_df[feature_columns].copy()

                    X = X.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    ).fillna(0)

                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)

                    result_df = input_df.copy()

                    result_df["Predicted Class"] = [
                        CLASS_NAMES.get(int(value), str(value))
                        for value in predictions
                    ]

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(X_scaled)

                        result_df["Confidence"] = (
                            probabilities.max(axis=1)
                        ).round(4)

                    st.success(
                        f"Successfully classified {len(result_df):,} records."
                    )

                    st.dataframe(
                        result_df.head(100),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Predictions",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="cyber_predictions.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# 8. FOOTER
# =========================================================

st.divider()

st.caption(
    "Adaptive AI-Based Cyber Defense System | "
    "MedSec-25 IoMT Dataset | Offline Experimental Dashboard"
)
