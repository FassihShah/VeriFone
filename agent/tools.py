from services.identify_phone_from_image import search_by_image
from services.predict_price_service import run_pipeline
from langchain_core.tools import Tool
from models import UsedMobile
from typing import Any, Optional
from dotenv import load_dotenv
import json
from pymongo import MongoClient
import os
import re

load_dotenv()

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
mobiles_collection = db["models_specs"]
model_names_collection = db["models_names"]



# Tools wrappers

def parse_input(x):
    """ Handles __arg1 string case, direct string case, and ensures final output is a dict.
    """
    if isinstance(x, dict) and "__arg1" in x and isinstance(x["__arg1"], str):
        try:
            x = json.loads(x["__arg1"])
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in __arg1: {x['__arg1']}")
    elif isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = {"model": x}
    if not isinstance(x, dict):
        raise ValueError(f"Expected dict, got {type(x)}")
    return x

def parse_input_image_tool(x):
    
    if isinstance(x, dict) and "__arg1" in x and isinstance(x["__arg1"], str):
        try:
            x = json.loads(x["__arg1"])
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in __arg1: {x['__arg1']}")
    elif isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = {"image_path": x}
    if not isinstance(x, dict):
        raise ValueError(f"Expected dict, got {type(x)}")
    return x


def image_understanding_wrapper(x):

    x = parse_input_image_tool(x)

    print(f"[DEBUG] Parsed input: {x} (type: {type(x)})")

    if "image_path" not in x:
        raise ValueError(f"Input must contain 'image_path' key, got keys: {list(x.keys())}")

    image_path = x["image_path"]

    print("Image path: ", image_path)

    return search_by_image(image_path)


def model_specs_wrapper(x):
    if isinstance(x, dict) and "__arg1" in x and isinstance(x["__arg1"], str):
        try:
            x = json.loads(x["__arg1"])
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in __arg1: {x['__arg1']}")
    elif isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            x = {"model": x}
    if not isinstance(x, dict):
        raise ValueError(f"Expected dict, got {type(x)}")

    model = x.get("model") or x.get("model_name")
    if not model:
        raise ValueError("Missing model name.")

    specs = get_specs_from_db(model)
    if not specs:
        return f"No specs found for model: {model}"

    return specs


def convert_fields(data: dict[str, Any]) -> dict[str, Any]:
    if "condition" in data:
        try:
            data["condition"] = int(data["condition"])
        except:
            raise ValueError("Field 'condition' must be an integer (e.g., 9)")

    if "pta_approved" in data:
        if isinstance(data["pta_approved"], str):
            data["pta_approved"] = data["pta_approved"].lower() in ["true", "yes", "1"]

    for key in [
        "is_panel_changed", "screen_crack", "panel_dot", "panel_line", "panel_shade",
        "camera_lens_ok", "fingerprint_ok", "with_box", "with_charger"
    ]:
        if key in data:
            if isinstance(data[key], str):
                data[key] = data[key].lower() in ["true", "yes", "1"]

    return data

def price_prediction_wrapper(x):
    x = parse_input(x)

    if "input_mobile" not in x:
        raise ValueError("Missing required field: 'input_mobile'")

    x["input_mobile"] = convert_fields(x["input_mobile"])
    
    validated_mobile = UsedMobile(**x["input_mobile"])

    predicted_price = run_pipeline(validated_mobile)
    return {"predicted_price": predicted_price}



# Tools


def get_specs_from_db(model: str) -> Optional[dict]:
    """
    Fetches mobile specs from MongoDB using the provided model name (case-insensitive).
    Does not require the brand; brand is ignored while matching.
    """
    if not model:
        raise ValueError("Model name is required.")

    # Search using regex to match model field case-insensitively
    result = mobiles_collection.find_one(
        {"model": {"$regex": f"^{re.escape(model)}$", "$options": "i"}}
    )

    if result:
        result.pop("_id", None)  # Remove MongoDB ID if present
        return result

    return None


image_understanding_tool = Tool(
    name="ImageUnderstandingAgent",
    description="""
    Use this tool when the user uploads an image of a mobile's back and wants to identify the brand and model.
    
    CRITICAL INPUT FORMAT:
    - Input must be a JSON object with `image_path` key
    - Use forward slashes (/) in paths, never backslashes (\)
    - Example: {"image_path": "C:/Users/dell/AppData/Local/Temp/image.jpg"}
    - Do NOT use nested JSON strings or double escaping
    
    Output: Detected model and brand.
    """,
    func=lambda x: image_understanding_wrapper(x)
)

model_specs_tool = Tool(
    name="ModelSpecsTool",
    description="""
    Use this tool when the user wants to retrieve the specs of a mobile phone that is already stored in the database.

    Input format: {"model": "Galaxy A52"}
    Output: Dictionary of specifications for the mobile model.
    """,
    func=lambda x: model_specs_wrapper(x)
)


price_prediction_tool = Tool(
    name="PricePredictionAgent",
    description="""
    Use this tool when the user wants to **predict the price** of a used mobile.

    Input should be a JSON object with a field `input_mobile` containing known details about the mobile phone.

    Required fields inside `input_mobile`:
    - model (e.g., "Hot 10")
    - condition (integer from 1 to 10)
    - pta_approved (true or false) ("cpid approved", "cpid", "pta" means its pta_approved = true )
      ("non pta", "sim lock", "jv" means its not approved i.e pta_approved = false)

    Optional fields that can also be provided (if known by user):
    - ram (e.g., "4GB")
    - storage (e.g., "64GB")
    - is_panel_changed, screen_crack, panel_dot, panel_line, panel_shade,
    - camera_lens_ok, fingerprint_ok,
    - with_box, with_charger

    Example:
    {
        "input_mobile": {
            "model": "Hot 10",
            "ram": "4GB",
            "storage": "64GB",
            "condition": 9,
            "pta_approved": true,
            "screen_crack": false,
            "with_charger": true
        }
    }

    The more complete the input, the more accurate the price prediction.

    Note:
    - If this tool returns need_fresh_data = True, it means we need fresh data to scrape from olx for that model. Then DataCollectorAgent should be used
    """,
    func=lambda x: price_prediction_wrapper(x)
)


tools = [
    image_understanding_tool,
    price_prediction_tool,
    model_specs_tool
]