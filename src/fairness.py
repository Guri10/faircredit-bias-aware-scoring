import pandas as pd
import numpy as np

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing

from src.preprocessing import load_data, split_and_preprocess

def run_reweighing(df: pd.DataFrame, priv_attr: str, priv_categories: list):
    """
    Apply AIF360’s Reweighing pre-processing on the raw training DataFrame.
    Returns a StandardDataset with .instance_weights to use in training.
    """
    # Prepare DataFrame for AIF360: numeric Risk, binary prot_attr
    df_rw = df[[priv_attr, "Risk"]].copy()
    df_rw["Risk"] = df_rw["Risk"].map({"good": 1, "bad": 0})
    df_rw[priv_attr] = df_rw[priv_attr].apply(lambda x: 1 if x in priv_categories else 0)

    dataset = StandardDataset(
        df=df_rw,
        label_name="Risk",
        favorable_classes=[1],
        protected_attribute_names=[priv_attr],
        privileged_classes=[[1]]
    )
    RW = Reweighing(
        unprivileged_groups=[{priv_attr: 0}],
        privileged_groups=[{priv_attr: 1}]
    )
    return RW.fit_transform(dataset)


def evaluate_fairness(
    X_test: np.ndarray,
    y_test: pd.Series,
    preprocessor,
    clf,
    df_test_raw: pd.DataFrame,
    prot_attr: str,
    priv_categories: list
):
    """
    Compute fairness metrics:
      - Statistical Parity Difference (manual)
      - Equal Opportunity Difference (manual)
      - Fairlearn: selection_rate & true_positive_rate by group

    Args:
        X_test: preprocessed features for test set
        y_test: numeric labels (1=good, 0=bad)
        preprocessor: fitted ColumnTransformer (unused here)
        clf: trained classifier
        df_test_raw: original raw DataFrame split for test (string codes)
        prot_attr: protected attribute column name
        priv_categories: list of privileged codes (e.g. ["A91","A93","A94"])

    Returns:
        aif_metrics: dict with 'stat_parity_diff' & 'eq_opp_diff'
        fairlearn_metrics: dict of group-level selection_rate & TPR
    """
    # Get binary predictions
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Group masks
    priv_mask = df_test_raw[prot_attr].isin(priv_categories).values
    unpriv_mask = ~priv_mask

    # Statistical Parity Difference
    sr_priv = preds[priv_mask].mean() if priv_mask.sum() > 0 else np.nan
    sr_unpriv = preds[unpriv_mask].mean() if unpriv_mask.sum() > 0 else np.nan
    stat_parity_diff = sr_unpriv - sr_priv

    # Equal Opportunity Difference (TPR diff)
    pos = (y_test.values == 1)
    tpr_priv = preds[priv_mask & pos].mean() if (priv_mask & pos).sum() > 0 else np.nan
    tpr_unpriv = preds[unpriv_mask & pos].mean() if (unpriv_mask & pos).sum() > 0 else np.nan
    equal_opportunity_diff = tpr_unpriv - tpr_priv

    aif_metrics = {
        "stat_parity_diff": stat_parity_diff,
        "eq_opp_diff": equal_opportunity_diff
    }

    # Fairlearn metrics by group
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate
        },
        y_true=y_test.values,
        y_pred=preds,
        sensitive_features=priv_mask.astype(int)
    )
    fairlearn_metrics = mf.by_group.to_dict()

    return aif_metrics, fairlearn_metrics


if __name__ == "__main__":
    # Quick smoke: load, split, train, eval fairness
    df = load_data()
    X_train, X_test, y_train, y_test, preproc = split_and_preprocess(df)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)
    # Define protected codes
    priv = ["A91", "A93", "A94"]
    aif, fl = evaluate_fairness(
        X_test, y_test, preproc, clf, 
        train_test_split(df, test_size=0.2, stratify=df['Risk'], random_state=42)[1],
        prot_attr="Personal_Status_Sex",
        priv_categories=priv
    )
    print("AIF360 metrics:", aif)
    print("Fairlearn metrics:", fl)
