# ============================================================
# UNIVERSAL PRODUCTION-GRADE CUSTOMER ANALYTICS PIPELINE
# ============================================================
# FEATURES
# ------------------------------------------------------------
# ✅ Smart universal column detection
# ✅ Exact + partial matching
# ✅ Duplicate-safe schema mapping
# ✅ Dynamic customer segmentation
# ✅ Dynamic campaign recommendation engine
# ✅ KMeans clustering
# ✅ RandomForest ML model
# ✅ SHAP explainability
# ✅ ROI simulation
# ✅ Robust validation
# ✅ Works on most retail/customer datasets
# ============================================================

# ============================================================
# IMPORTS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import shap

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "retail_india_customers_final.csv"

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully")
print(f"Dataset Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# SMART UNIVERSAL COLUMN DETECTION
# ============================================================

COLUMN_PATTERNS = {

    "spend": [
        "monthly_spend",
        "total_spend",
        "avg_spend",
        "spend",
        "amount",
        "sales",
        "revenue"
    ],

    "frequency": [
        "purchase_frequency",
        "shopping_frequency",
        "frequency",
        "visit",
        "transaction",
        "orders"
    ],

    "basket": [
        "basket_size",
        "basket",
        "cart",
        "quantity",
        "items"
    ],

    "promo": [
        "promo_response",
        "campaign_response",
        "promo",
        "response"
    ],

    "age": [
        "age_group",
        "age"
    ],

    "gender": [
        "gender",
        "sex"
    ],

    "city": [
        "city_tier",
        "city",
        "location",
        "region"
    ],

    "payment": [
        "payment_method",
        "payment"
    ],

    "product": [
        "product_category",
        "product",
        "category",
        "item"
    ]
}


def detect_columns(df):

    detected = {}
    used_columns = set()

    columns_lower = {
        col.lower(): col
        for col in df.columns
    }

    # --------------------------------------------------------
    # EXACT MATCHING
    # --------------------------------------------------------

    for feature, patterns in COLUMN_PATTERNS.items():

        for pattern in patterns:

            for lower_col, original_col in columns_lower.items():

                if original_col in used_columns:
                    continue

                if lower_col == pattern:

                    detected[feature] = original_col
                    used_columns.add(original_col)
                    break

            if feature in detected:
                break

    # --------------------------------------------------------
    # PARTIAL MATCHING FALLBACK
    # --------------------------------------------------------

    for feature, patterns in COLUMN_PATTERNS.items():

        if feature in detected:
            continue

        for pattern in patterns:

            for lower_col, original_col in columns_lower.items():

                if original_col in used_columns:
                    continue

                if pattern in lower_col:

                    detected[feature] = original_col
                    used_columns.add(original_col)
                    break

            if feature in detected:
                break

    return detected


cols = detect_columns(df)

print("\nDetected Columns:")
for k, v in cols.items():
    print(f"{k} --> {v}")

# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "spend",
    "basket",
    "frequency"
]

missing = [
    col for col in required_columns
    if col not in cols
]

if missing:

    raise ValueError(
        f"\nMissing Required Columns: {missing}"
    )

print("\nValidation Passed")

# ============================================================
# DATA CLEANING
# ============================================================

print("\nCleaning dataset...")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Fill numeric missing values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

print("Data Cleaning Completed")

# ============================================================
# AGE HANDLING
# ============================================================

age_group_col = None

if "age" in cols:

    age_col = cols["age"]

    # Numeric age
    if pd.api.types.is_numeric_dtype(df[age_col]):

        df["Age_Group_Binned"] = pd.cut(
            df[age_col],
            bins=[0, 25, 35, 50, 100],
            labels=[
                "18-25",
                "26-35",
                "36-50",
                "50+"
            ]
        )

        age_group_col = "Age_Group_Binned"

    else:

        age_group_col = age_col

# ============================================================
# PROMO RESPONSE NORMALIZATION
# ============================================================

def normalize_promo(value):

    value = str(value).strip().lower()

    positive_values = [
        "yes",
        "y",
        "1",
        "true",
        "responded",
        "accepted"
    ]

    return 1 if value in positive_values else 0


if "promo" in cols:

    df["Promo_Response_Normalized"] = (
        df[cols["promo"]]
        .apply(normalize_promo)
    )

else:

    df["Promo_Response_Normalized"] = 0

# ============================================================
# DYNAMIC GROUPING
# ============================================================

group_cols = []

if "city" in cols:
    group_cols.append(cols["city"])

if age_group_col:
    group_cols.append(age_group_col)

if "gender" in cols:
    group_cols.append(cols["gender"])

# Fallback
if len(group_cols) == 0:
    group_cols = [df.index]

print("\nGrouping Columns:")
print(group_cols)

# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

print("\nCreating customer segments...")

aggregation_dict = {}

aggregation_dict[cols["spend"]] = "mean"
aggregation_dict[cols["basket"]] = "mean"
aggregation_dict[cols["frequency"]] = "mean"

aggregation_dict[
    "Promo_Response_Normalized"
] = "mean"

segments = (
    df.groupby(group_cols)
    .agg(aggregation_dict)
    .reset_index()
)

# ============================================================
# SAFE RENAME
# ============================================================

rename_dict = {

    cols["spend"]: "Avg_Spend",

    cols["basket"]: "Avg_Basket",

    cols["frequency"]: "Avg_Frequency",

    "Promo_Response_Normalized":
        "Promo_Response_Rate"
}

segments.rename(
    columns=rename_dict,
    inplace=True
)

print("\nSegment Columns:")
print(segments.columns.tolist())

# ============================================================
# VALIDATE SEGMENTS
# ============================================================

required_metrics = [
    "Avg_Spend",
    "Avg_Basket",
    "Avg_Frequency",
    "Promo_Response_Rate"
]

missing_metrics = [
    col for col in required_metrics
    if col not in segments.columns
]

if missing_metrics:

    raise ValueError(
        f"\nMissing Aggregated Metrics: {missing_metrics}"
    )

print("\nSegment Validation Passed")

# ============================================================
# DYNAMIC THRESHOLDS
# ============================================================

print("\nCalculating Dynamic Thresholds...")

high_spend = (
    segments["Avg_Spend"]
    .quantile(0.75)
)

high_basket = (
    segments["Avg_Basket"]
    .quantile(0.75)
)

high_frequency = (
    segments["Avg_Frequency"]
    .quantile(0.75)
)

high_response = (
    segments["Promo_Response_Rate"]
    .quantile(0.75)
)

# ============================================================
# SMART CAMPAIGN ENGINE
# ============================================================

def recommend_campaign(row):

    if (
        row["Avg_Spend"] >= high_spend and
        row["Promo_Response_Rate"] >= high_response
    ):

        return "Premium Loyalty Campaign"

    elif (
        row["Avg_Basket"] >= high_basket
    ):

        return "Bundle Campaign"

    elif (
        row["Avg_Frequency"] >= high_frequency
    ):

        return "Repeat Customer Rewards"

    elif (
        row["Promo_Response_Rate"] >= 0.30
    ):

        return "Festival Campaign"

    else:

        return "Discount Campaign"


segments["Recommended_Campaign"] = (
    segments.apply(
        recommend_campaign,
        axis=1
    )
)

print("Campaign Recommendation Completed")

# ============================================================
# ROI SIMULATION
# ============================================================

segments["Baseline_Revenue"] = (
    segments["Avg_Spend"] *
    segments["Avg_Frequency"]
)

segments["Expected_Uplift"] = (
    1 +
    (
        segments["Promo_Response_Rate"] * 0.30
    )
)

segments["Projected_Revenue"] = (
    segments["Baseline_Revenue"] *
    segments["Expected_Uplift"]
)

# ============================================================
# KMEANS CLUSTERING
# ============================================================

print("\nRunning KMeans Clustering...")

cluster_features = segments[
    [
        "Avg_Spend",
        "Avg_Basket",
        "Avg_Frequency"
    ]
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    cluster_features
)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

segments["Customer_Cluster"] = (
    kmeans.fit_predict(scaled_features)
)

print("KMeans Clustering Completed")

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining RandomForest Model...")

campaign_mapping = {
    campaign: idx
    for idx, campaign in enumerate(
        segments["Recommended_Campaign"]
        .unique()
    )
}

segments["Campaign_Label"] = (
    segments["Recommended_Campaign"]
    .map(campaign_mapping)
)

X = segments[
    [
        "Avg_Spend",
        "Avg_Basket",
        "Avg_Frequency",
        "Promo_Response_Rate"
    ]
]

y = segments["Campaign_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\nGenerating SHAP Summary Plot...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Saved")

# ============================================================
# VISUALIZATION
# ============================================================

sns.set_style("whitegrid")

# ------------------------------------------------------------
# CAMPAIGN DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    y=segments["Recommended_Campaign"]
)

plt.title(
    "Recommended Campaign Distribution"
)

plt.tight_layout()

plt.savefig(
    "campaign_distribution.png"
)

plt.close()

# ------------------------------------------------------------
# CUSTOMER CLUSTERS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=segments,
    x="Avg_Spend",
    y="Avg_Frequency",
    hue="Customer_Cluster",
    s=120
)

plt.title("Customer Clusters")

plt.tight_layout()

plt.savefig(
    "customer_clusters.png"
)

plt.close()

# ------------------------------------------------------------
# ROI SIMULATION
# ------------------------------------------------------------

roi_df = pd.DataFrame({

    "Revenue_Type": [
        "Baseline Revenue",
        "Projected Revenue"
    ],

    "Revenue": [
        segments["Baseline_Revenue"].sum(),
        segments["Projected_Revenue"].sum()
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=roi_df,
    x="Revenue_Type",
    y="Revenue"
)

plt.title(
    "Revenue Uplift Simulation"
)

plt.tight_layout()

plt.savefig(
    "roi_simulation.png"
)

plt.close()

# ============================================================
# EXPORT RESULTS
# ============================================================

OUTPUT_FILE = (
    "production_customer_segments.csv"
)

segments.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n================================================")
print("PIPELINE EXECUTED SUCCESSFULLY")
print("================================================")

print("\nGenerated Files:")
print(f"1. {OUTPUT_FILE}")
print("2. shap_summary.png")
print("3. campaign_distribution.png")
print("4. customer_clusters.png")
print("5. roi_simulation.png")

print("\nSample Output:")
print(
    segments.head()
)

# ============================================================
# END OF PIPELINE
# ============================================================
