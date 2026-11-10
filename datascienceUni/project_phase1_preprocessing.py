# %% [markdown]
# # Data Science — Project Phase 1: Data Preprocessing
# **Hashemite University – Faculty of Engineering – Department of Computer Engineering**
# Course: Data Science (2104081410) · Instructor: Dr. Mohammad Al-Hammouri · Fall 2026/2027
# 
# Dataset: `Project_dataset.csv` (customer data, 20,330 rows × 23 columns).
# 
# Pipeline: **load → explore → fix types/formats → standardize text → fix dates → remove duplicates → handle invalid values →
# handle missing values → handle outliers → feature engineering → encoding → correlation/feature selection → scaling → export**.

# %% [markdown]
# ## 0. Setup

# %%
import os, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")

DATA_PATH = os.path.join("data", "Project_dataset.csv")
FIG_DIR = os.path.join("outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 1. Load the data and first look

# %%
raw = pd.read_csv(DATA_PATH)
df = raw.copy()
print("Shape:", df.shape)
df.head()

# %%
df.info()

# %%
print(df.describe(include="all").T)

# %% [markdown]
# ## 2. Data quality assessment (before cleaning)

# %%
def quality_report(d):
    return pd.DataFrame({
        "dtype": d.dtypes.astype(str),
        "missing": d.isna().sum(),
        "missing_%": (d.isna().mean() * 100).round(2),
        "unique": d.nunique(),
    })

qr_before = quality_report(df)
print("Exact duplicate rows      :", df.duplicated().sum())
print("Duplicated client_ref IDs :", df["client_ref"].duplicated().sum())
print(qr_before)

# %%
# Missing values per column (before)
miss = df.isna().sum().sort_values(ascending=False)
miss = miss[miss > 0]
plt.figure(figsize=(9, 4))
sns.barplot(x=miss.values, y=miss.index, color="#4C72B0")
plt.title("Missing values per column (before cleaning)")
plt.xlabel("Number of missing values"); plt.ylabel("")
savefig("01_missing_before.png")

# %%
# Inconsistent categories: the same value written in many ways
for col in ["sex", "education", "item_category", "satisfaction_level",
            "account_status", "previous_churn_flag"]:
    print(f"{col:20s} ->", df[col].value_counts(dropna=False).to_dict())

# %%
# Date formats found (digits replaced with 9)
for col in ["registration_date", "birth_date"]:
    pats = df[col].dropna().map(lambda s: re.sub(r"[A-Za-z]{3}", "Mon", re.sub(r"\d", "9", s)))
    print(col, pats.value_counts().to_dict())

# %% [markdown]
# **Issues found:**
# 1. `spend_per_order` is text: some values have a `$` sign, thousands separators and spaces.
# 2. Text columns use different capitalisation and have extra spaces (`male`, `MALE`, `  Male `).
# 3. Yes/No columns are written in 10 ways (`Yes, yes, Y, 1, TRUE / No, no, N, 0, FALSE`).
# 4. Dates come in 4 formats (`m/d/Y`, `d/m/Y`, `dd-Mon-yy`, `Y-m-d`).
# 5. Placeholder values: `?` and `unknown` in `satisfaction_level`; `-999, -100, 0, 999` in `credit_rating_score`.
# 6. Duplicate rows, and the same customer stored more than once with small differences.
# 7. Missing values in 11 columns.
# 8. Outliers in `order_count` (values ×20 the normal range), `yearly_salary`, `monthly_salary`.
# 9. Columns with no useful information (`record_status` has one value; IDs and names).

# %% [markdown]
# ## 3. Fix data types and standardize text

# %%
# 3.1 strip spaces from every text column; turn empty strings into NaN
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df[c] = df[c].str.strip().replace("", np.nan)

# 3.2 consistent capitalisation
df["client_name"]   = df["client_name"].str.title()
df["sex"]           = df["sex"].str.capitalize()
df["location"]      = df["location"].str.title()
df["item_category"] = df["item_category"].str.title()
df["country_iso"]   = df["country_iso"].str.upper()
edu_map = {"high school": "High School", "bachelors": "Bachelors", "masters": "Masters", "phd": "PhD"}
df["education"] = df["education"].str.lower().map(edu_map)

# 3.3 placeholder text -> NaN
df["satisfaction_level"] = df["satisfaction_level"].replace({"?": np.nan, "unknown": np.nan})
df["salary_bracket"]     = df["salary_bracket"].replace({"Unknown": np.nan})

# 3.4 unify boolean columns to 1 / 0
bool_map = {"yes": 1, "y": 1, "1": 1, "true": 1, "no": 0, "n": 0, "0": 0, "false": 0}
for c in ["account_status", "previous_churn_flag", "premium_customer"]:
    df[c] = df[c].astype(str).str.lower().map(bool_map).astype("Int64")

# 3.5 money text -> float
df["spend_per_order"] = pd.to_numeric(
    df["spend_per_order"].str.replace(r"[$,\s]", "", regex=True), errors="coerce")

for col in ["sex", "location", "education", "item_category", "satisfaction_level",
            "account_status", "previous_churn_flag", "premium_customer", "country_iso"]:
    print(f"{col:20s} ->", df[col].value_counts(dropna=False).to_dict())
print("spend_per_order dtype:", df["spend_per_order"].dtype, "| NaN:", df["spend_per_order"].isna().sum())

# %% [markdown]
# ## 4. Parse dates (mixed formats)

# %%
def parse_date(s):
    """Parse the 4 date formats in this file. m/d/Y is the main format;
    when the first number is > 12 the value can only be d/m/Y."""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if re.fullmatch(r"\d{1,2}-[A-Za-z]{3}-\d{2}", s):
        return pd.to_datetime(s, format="%d-%b-%y", errors="coerce")
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        a = int(m.group(1))
        fmt = "%d/%m/%Y" if a > 12 else "%m/%d/%Y"
        return pd.to_datetime(s, format=fmt, errors="coerce")
    return pd.NaT

for c in ["registration_date", "birth_date"]:
    before_na = df[c].isna().sum()
    df[c] = df[c].map(parse_date)
    print(f"{c}: missing before = {before_na}, missing after parse = {df[c].isna().sum()}, "
          f"range = {df[c].min().date()} -> {df[c].max().date()}")

# %% [markdown]
# ## 5. Remove duplicates

# %%
n0 = len(df)
df = df.drop_duplicates()
n1 = len(df)
# Same customer (client_ref) stored more than once, e.g. with a different web session.
# After standardizing the text these records are the same person -> keep the first one.
df = df.drop_duplicates(subset="client_ref", keep="first")
n2 = len(df)
print(f"Exact duplicate rows removed        : {n0 - n1}")
print(f"Repeated client_ref records removed : {n1 - n2}")
print(f"Rows left                           : {n2}")
assert df["client_ref"].is_unique

# %% [markdown]
# ## 6. Handle invalid values (domain rules)

# %%
REF_DATE = pd.Timestamp("2024-12-31")   # last registration date in the data

# credit score must be in the FICO range 300..850; -999 / -100 / 0 / 999 are placeholders
bad_credit = ~df["credit_rating_score"].between(300, 850) & df["credit_rating_score"].notna()
print("Invalid credit scores ->", df.loc[bad_credit, "credit_rating_score"].value_counts().to_dict())
df.loc[bad_credit, "credit_rating_score"] = np.nan

# birth date: not in the future and customer at least 18 and at most 100 years old at REF_DATE
age = (REF_DATE - df["birth_date"]).dt.days / 365.25
bad_birth = df["birth_date"].notna() & ~age.between(18, 100)
print("Invalid birth dates (future / age < 18 / age > 100):", bad_birth.sum())
df.loc[bad_birth, "birth_date"] = pd.NaT

# order_count: normal values are 1..49; the large values are all multiples of 20 (data entry error)
bad_orders = df["order_count"] > 50
print("Invalid order_count values:", bad_orders.sum(),
      "| all multiples of 20:", bool((df.loc[bad_orders, "order_count"] % 20 == 0).all()))
df.loc[bad_orders, "order_count"] = np.nan

# satisfaction_points outside 0..100 ?
print("satisfaction_points outside 0..100:", (~df["satisfaction_points"].between(0, 100) & df["satisfaction_points"].notna()).sum())

# %% [markdown]
# ## 7. Handle missing values

# %%
print(df.isna().sum()[df.isna().sum() > 0])

# %%
# 7.1 yearly_salary: monthly_salary is never missing and yearly = 12 x monthly (corr = 0.99999)
m = df["yearly_salary"].isna()
df.loc[m, "yearly_salary"] = df.loc[m, "monthly_salary"] * 12
print("yearly_salary filled from monthly_salary:", m.sum())

# 7.2 salary_bracket: recompute from salary using the bracket limits seen in the data
df["salary_bracket"] = pd.cut(df["yearly_salary"], bins=[0, 25000, 50000, 100000, np.inf],
                              labels=["Low", "Mid", "High", "Very High"], right=False).astype(str)

# 7.3 satisfaction_level <-> satisfaction_points are two forms of the same thing
sat_bins   = [-np.inf, 19, 39, 60, 80, np.inf]
sat_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
print(df.groupby("satisfaction_level")["satisfaction_points"].agg(["min", "max", "mean"]).round(1))

m = df["satisfaction_level"].isna() & df["satisfaction_points"].notna()
df.loc[m, "satisfaction_level"] = pd.cut(df.loc[m, "satisfaction_points"], sat_bins, labels=sat_labels).astype(str)
print("satisfaction_level filled from points:", m.sum())

m = df["satisfaction_points"].isna() & df["satisfaction_level"].notna()
df.loc[m, "satisfaction_points"] = df.loc[m, "satisfaction_level"].map(
    df.groupby("satisfaction_level")["satisfaction_points"].median())
print("satisfaction_points filled with level median:", m.sum())

both = df["satisfaction_points"].isna()
df.loc[both, "satisfaction_points"] = df["satisfaction_points"].median()
df.loc[both, "satisfaction_level"] = df["satisfaction_level"].mode()[0]
print("both missing -> median / mode:", both.sum())

# %%
# 7.4 numeric columns -> median (robust to outliers)
for c in ["credit_rating_score", "tenure_years", "order_count"]:
    n = df[c].isna().sum()
    df[c] = df[c].fillna(df[c].median())
    print(f"{c:20s} filled with median {df[c].median():.1f} ({n} values)")

# 7.5 birth_date -> median birth date
n = df["birth_date"].isna().sum()
df["birth_date"] = df["birth_date"].fillna(df["birth_date"].median())
print(f"birth_date filled with median date ({n} values)")

# 7.6 categorical columns
n = df["education"].isna().sum()
df["education"] = df["education"].fillna(df["education"].mode()[0])
print(f"education filled with mode '{df['education'].mode()[0]}' ({n} values)")

# location and country are missing together (same 609 rows) -> separate 'Unknown' category
n = df["location"].isna().sum()
df["location"] = df["location"].fillna("Unknown")
df["country_iso"] = df["country_iso"].fillna("Unknown")
print(f"location / country_iso filled with 'Unknown' ({n} values)")

# client_name is not used for modelling -> placeholder
df["client_name"] = df["client_name"].fillna("Unknown")

print("\nMissing values left:", int(df.isna().sum().sum()))

# %% [markdown]
# ## 8. Handle outliers

# %%
num_cols = ["yearly_salary", "monthly_salary", "spend_per_order", "credit_rating_score",
            "tenure_years", "order_count", "satisfaction_points"]

def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

print(pd.DataFrame({c: {"IQR outliers": iqr_outliers(df[c]),
                        "skew": round(df[c].skew(), 2)} for c in num_cols}).T)

fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 4))
for ax, c in zip(axes, num_cols):
    sns.boxplot(y=df[c], ax=ax, color="#DD8452"); ax.set_title(c, fontsize=9); ax.set_ylabel("")
plt.suptitle("Box plots before outlier treatment")
savefig("02_boxplots_before.png")

# %%
# Salaries and spend are right-skewed (real high earners, not errors: yearly = 12 x monthly still holds).
# Instead of deleting customers we cap (winsorize) at the 1st / 99th percentile.
caps = {}
for c in ["yearly_salary", "monthly_salary", "spend_per_order"]:
    lo, hi = df[c].quantile([0.01, 0.99])
    caps[c] = (round(lo, 2), round(hi, 2))
    df[c] = df[c].clip(lo, hi)
print("Capping limits:", caps)

fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 4))
for ax, c in zip(axes, num_cols):
    sns.boxplot(y=df[c], ax=ax, color="#55A868"); ax.set_title(c, fontsize=9); ax.set_ylabel("")
plt.suptitle("Box plots after outlier treatment")
savefig("03_boxplots_after.png")

# %% [markdown]
# ## 9. Feature engineering

# %%
df["age"] = ((REF_DATE - df["birth_date"]).dt.days / 365.25).round().astype(int)
df["registration_year"] = df["registration_date"].dt.year
df["days_since_registration"] = (REF_DATE - df["registration_date"]).dt.days
df["total_spend"] = (df["spend_per_order"] * df["order_count"]).round(2)
df["log_yearly_salary"] = np.log1p(df["yearly_salary"])

print(df[["age", "registration_year", "days_since_registration", "total_spend", "log_yearly_salary"]].describe().T.round(2))

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axes.ravel(), ["age", "yearly_salary", "log_yearly_salary", "spend_per_order",
                                "credit_rating_score", "order_count", "satisfaction_points", "total_spend"]):
    sns.histplot(df[c], bins=40, ax=ax, color="#4C72B0"); ax.set_title(c)
plt.suptitle("Distributions after cleaning")
savefig("04_distributions_after.png")

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, c in zip(axes, ["sex", "education", "item_category", "satisfaction_level"]):
    order = df[c].value_counts().index
    sns.countplot(x=df[c], order=order, ax=ax, color="#8172B3"); ax.set_title(c); ax.tick_params(axis="x", rotation=30)
plt.suptitle("Categorical columns after standardization")
savefig("05_categories_after.png")

# %% [markdown]
# ## 10. Encoding categorical variables

# %%
clean = df.copy()   # human-readable cleaned dataset (saved later)

enc = df.copy()
# ordinal columns -> ordered integers
enc["education"]          = enc["education"].map({"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3})
enc["satisfaction_level"] = enc["satisfaction_level"].map({"Very Low": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4})
enc["salary_bracket"]     = enc["salary_bracket"].map({"Low": 0, "Mid": 1, "High": 2, "Very High": 3})
# binary
enc["sex"] = enc["sex"].map({"Male": 1, "Female": 0})
# nominal -> one-hot
enc = pd.get_dummies(enc, columns=["location", "item_category"], prefix=["loc", "cat"], dtype=int)
print("Encoded shape:", enc.shape)
print(enc.head())

# %% [markdown]
# ## 11. Correlation analysis and feature selection

# %%
corr_cols = ["yearly_salary", "monthly_salary", "log_yearly_salary", "salary_bracket", "spend_per_order",
             "credit_rating_score", "tenure_years", "order_count", "total_spend", "satisfaction_points",
             "satisfaction_level", "education", "age", "days_since_registration", "registration_year",
             "account_status", "previous_churn_flag", "premium_customer", "sex"]
corr = enc[corr_cols].astype(float).corr()
plt.figure(figsize=(13, 10))
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, square=True)
plt.title("Correlation matrix")
savefig("06_correlation.png")

pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
         .rename("corr").reset_index().assign(abs=lambda d: d["corr"].abs())
         .sort_values("abs", ascending=False))
print("Highly correlated pairs (|r| > 0.85):")
print(pairs[pairs["abs"] > 0.85][["level_0", "level_1", "corr"]].round(3).to_string(index=False))

# %%
drop_cols = [
    "client_ref",          # ID only
    "client_name",         # personal identifier, no predictive value
    "web_session_ref",     # technical ID
    "record_status",       # constant (one value)
    "country_iso",         # fully determined by location
    "monthly_salary",      # = yearly_salary / 12 (r = 1.00)
    "yearly_salary",       # replaced by its log version
    "salary_bracket",      # binned copy of salary
    "satisfaction_level",  # binned copy of satisfaction_points
    "birth_date", "registration_date",   # replaced by age / days_since_registration
    "registration_year",   # same info as days_since_registration
]
print("Constant columns:", [c for c in raw.columns if raw[c].nunique() == 1])
model_df = enc.drop(columns=drop_cols)
print("Model-ready shape:", model_df.shape)
print(list(model_df.columns))

# %% [markdown]
# ## 12. Train / test split and feature scaling

# %%
# The scaler is fitted on the training set only, to avoid data leakage.
# previous_churn_flag is used as an example target for the next phase.
TARGET = "previous_churn_flag"
X = model_df.drop(columns=[TARGET])
y = model_df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scale_cols = ["log_yearly_salary", "spend_per_order", "credit_rating_score", "tenure_years",
              "order_count", "total_spend", "satisfaction_points", "education", "age",
              "days_since_registration"]
scaler = StandardScaler()
X_train = X_train.copy(); X_test = X_test.copy()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Target balance (train):", y_train.value_counts(normalize=True).round(3).to_dict())
print(X_train[scale_cols].describe().T[["mean", "std"]].round(3))

# %% [markdown]
# ## 13. Summary and export

# %%
qr_after = quality_report(clean)
summary = pd.DataFrame({
    "Rows":            [len(raw), len(clean)],
    "Columns":         [raw.shape[1], clean.shape[1]],
    "Missing values":  [int(raw.isna().sum().sum()), int(clean.isna().sum().sum())],
    "Duplicate rows":  [int(raw.duplicated().sum()), int(clean.duplicated().sum())],
    "Duplicate IDs":   [int(raw["client_ref"].duplicated().sum()), int(clean["client_ref"].duplicated().sum())],
}, index=["Before", "After"]).T
print(summary)

out = "outputs"
clean.to_csv(os.path.join(out, "customers_cleaned.csv"), index=False)
model_df.to_csv(os.path.join(out, "customers_model_ready.csv"), index=False)
pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(out, "train_scaled.csv"), index=False)
pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(out, "test_scaled.csv"), index=False)
print("\nSaved files:", sorted(os.listdir(out)))

