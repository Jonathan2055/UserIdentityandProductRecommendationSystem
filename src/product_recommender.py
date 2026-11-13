"""Bridge between a recognised team member and Jonathan's product model.

Team members are not customers in the transactions data, so config.MEMBER_TO_CUSTOMER
links each member to one real customer_id. We take that customer's most recent
transaction row from merged_dataset_encoded.csv (same columns the model was trained
on) and ask the product model which category they are likely to buy.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def load_product_model() -> dict:
    return joblib.load(C.PRODUCT_MODEL_PATH)   # {"model", "features", "classes"}


def customer_profile(member: str) -> pd.Series:
    cid = C.MEMBER_TO_CUSTOMER[member]
    data = pd.read_csv(C.MERGED_ENCODED_CSV)
    rows = data[data["customer_id"] == cid]
    if rows.empty:
        raise KeyError(f"customer_id {cid} (linked to {member}) not found in {C.MERGED_ENCODED_CSV.name}")
    return rows.sort_values("transaction_id").iloc[-1]       # latest purchase = current state


def recommend(member: str, bundle: dict | None = None) -> dict:
    bundle = bundle or load_product_model()
    row = customer_profile(member)
    X = row[bundle["features"]].to_frame().T.astype(float)
    proba = bundle["model"].predict_proba(X)[0]
    ranking = sorted(zip(bundle["model"].classes_, proba), key=lambda t: -t[1])
    return {"customer_id": int(row["customer_id"]), "top": ranking[0][0], "ranking": ranking,
            "profile": {"purchases": int(row["customer_n_purchases"]),
                        "avg_spend": float(row["customer_avg_spend"]),
                        "avg_engagement": float(row["avg_engagement"]),
                        "avg_purchase_interest": float(row["avg_purchase_interest"])}}
