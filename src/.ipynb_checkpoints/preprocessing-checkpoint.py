import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(proj_root, "data", "raw", "german.data")


# 1. Column names (same as your notebook)
col_names = [
    "Status_Checking_Account", "Duration", "Credit_History", "Purpose",
    "Credit_Amount", "Savings_Account", "Present_Employment",
    "Installment_Rate", "Personal_Status_Sex", "Other_Debtors",
    "Present_Residence_Since", "Property", "Age", "Other_Installment_Plans",
    "Housing", "Number_of_Existing_Credits", "Job",
    "Number_of_People_Liable", "Telephone", "Foreign_Worker", "Risk"
]

def load_data(path="../data/raw/german.data"):
    """Load the UCI German Credit data and map the target."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=col_names,
        engine="python"
    )
    df["Risk"] = df["Risk"].map({1: "good", 2: "bad"})
    return df

def build_preprocessor(df: pd.DataFrame):
    """Construct a ColumnTransformer for numeric & categorical features."""
    X = df.drop("Risk", axis=1)
    # identify column types
    num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_feats = X.select_dtypes(include=["object"]).columns.tolist()

    # numeric: median impute + standard scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # categorical: constant “missing” + one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_feats),
        ("cat", cat_pipeline, cat_feats),
    ])
    return preprocessor

def split_and_preprocess(df, test_size=0.2, random_state=42):
    """Split into train/test, fit the preprocessor on train, transform both."""
    df = df.copy()
    # numeric target encoding
    df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})

    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    preprocessor = build_preprocessor(df)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor

if __name__ == "__main__":
    # quick smoke-test
    df = load_data()
    X_tr, X_te, y_tr, y_te, pp = split_and_preprocess(df)
    print("✓ Preprocessing pipeline works")
    print("Train shape:", X_tr.shape, "Test shape:", X_te.shape)
