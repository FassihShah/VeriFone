from typing import List
from datetime import datetime, timedelta, timezone
from pymongo.collection import Collection
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import re
import os
from pymongo import MongoClient

from models import UsedMobile

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")

DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

MAX_DATA_AGE_DAYS = 30


def fetch_training_data(input_model: str, db: Collection = collection) -> List[UsedMobile]:
    """Fetches mobiles matching model and extracted in last 30 days"""

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=MAX_DATA_AGE_DAYS)

    query = {
        "model": {
            "$regex": re.escape(input_model),
            "$options": "i"
        },
        "extraction_date": {
            "$gte": cutoff_date
        }
    }

    training_data = []
    result = db.find(query)

    for doc in result:
        try:
            if "images" in doc and isinstance(doc["images"], str):
                doc["images"] = [img.strip() for img in doc["images"].split(",") if img.strip()]
            training_data.append(UsedMobile(**doc))
        except Exception as e:
            print(f"Skipping record: {e}")

    # Check if we have enough fresh records
    if len(training_data) < 50:
        raise RuntimeError(f"⚠️ Only {len(training_data)} fresh records found. Minimum 100 required.")

    return training_data



def preprocess_input_mobile(input_mobile: UsedMobile) -> pd.DataFrame:
    input_dict = input_mobile.model_dump()

    for field in ["ram", "storage"]:
        val = input_dict.get(field)
        if isinstance(val, str) and "GB" in val.upper():
            numeric_part = ''.join(filter(str.isdigit, val))
            input_dict[field] = int(numeric_part) if numeric_part else None

    for key, value in input_dict.items():
        if isinstance(value, bool):
            input_dict[key] = int(value)

    df = pd.DataFrame([input_dict])
    df.drop(columns=[col for col in ["price", "images", "post_date", "listing_source", "city"] if col in df.columns], inplace=True)

    return df


def preprocess_training_data(training_data: List[UsedMobile]) -> pd.DataFrame:
    processed = []
    fallback_ram = fallback_storage = None

    for item in training_data:
        if item.ram and "GB" in item.ram and item.storage and "GB" in item.storage:
            fallback_ram = item.ram
            fallback_storage = item.storage
            break

    if not fallback_ram or not fallback_storage:
        raise ValueError("No fallback RAM or Storage")

    for item in training_data:
        row = item.model_dump()
        row["ram"] = row.get("ram") or fallback_ram
        row["storage"] = row.get("storage") or fallback_storage

        for field in ["ram", "storage"]:
            val = row.get(field)
            if isinstance(val, str):
                match = re.search(r'\d+', val)
                row[field] = int(match.group()) if match else 6
            elif not isinstance(val, int):
                row[field] = 6

        for key, value in row.items():
            if isinstance(value, bool):
                row[key] = int(value)

        processed.append(row)

    df = pd.DataFrame(processed)
    df.drop(columns=["images", "post_date", "listing_source", "city", "model", "brand"], inplace=True, errors="ignore")

    return df


def train_model(training_df: pd.DataFrame) -> RandomForestRegressor:
    df = training_df.dropna(subset=["price"])
    X = df.drop(columns=["price"])
    y = df["price"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def predict_price(model: RandomForestRegressor, input_df: pd.DataFrame, mobile: UsedMobile) -> int:
    df = input_df.copy()
    df.drop(columns=["model", "brand"], inplace=True, errors="ignore")

    predicted_price = model.predict(df)[0]

    if mobile.is_panel_changed:
        predicted_price *= 0.8
    if mobile.panel_dot:
        predicted_price *= 0.75
    if mobile.panel_line:
        predicted_price *= 0.7
    if mobile.panel_shade:
        predicted_price *= 0.75
    if mobile.screen_crack:
        predicted_price *= 0.7
    if mobile.camera_lens_ok is False:
        predicted_price *= 0.9
    if mobile.fingerprint_ok is False:
        predicted_price *= 0.85
    if mobile.pta_approved is False:
        predicted_price *= 0.8

    return round(predicted_price / 500) * 500



def run_pipeline(input_mobile: UsedMobile, db: Collection = collection) -> int:
    """
    Runs the full price prediction pipeline on the given mobile.
    Returns the final predicted price.
    """

    training_data = fetch_training_data(input_mobile.model, db)

    if not training_data:
        raise RuntimeError("No training data found for the given model.")

    input_df = preprocess_input_mobile(input_mobile)
    training_df = preprocess_training_data(training_data)

    model = train_model(training_df)

    predicted_price = predict_price(model, input_df, input_mobile)

    return predicted_price

input_mobile = UsedMobile(
    brand="Google",
    model="Pixel 6A",
    ram="4GB",
    storage="128GB",
    is_panel_changed=False,
    panel_dot=False,
    panel_line=False,
    panel_shade=False,
    screen_crack=False,
    camera_lens_ok=True,
    fingerprint_ok=True,
    pta_approved=True,
    price=None
)

# print(run_pipeline(input_mobile,collection))

