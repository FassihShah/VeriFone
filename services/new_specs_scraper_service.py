from typing import List, Optional
from bs4 import BeautifulSoup
import requests
import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import time
from models import NewMobile

load_dotenv()

MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
mobiles_collection = db["models_specs"]
model_names_collection = db["models_names"]


def parse_ram_and_storage(mixed_string: str):

    ram_set = set()
    storage_set = set()
    matches = re.findall(r'(\d+GB)\s+(\d+GB)\s+RAM', mixed_string)

    for storage, ram in matches:
        storage_set.add(storage)
        ram_set.add(ram)

    return sorted(storage_set), sorted(ram_set)



def convert_specs_to_mobile(specs: dict) -> NewMobile:

    def get(*keys: str) -> Optional[str]:
        for key in keys:
            if key in specs:
                return specs[key]
        return None

    def extract_year_from_release(release_info: Optional[str]) -> Optional[int]:
        if release_info:
            for token in release_info.split():
                if token.isdigit() and len(token) == 4:
                    return int(token)
        return None

    mobile = NewMobile(
        brand=None,
        model=None,
        os=get("Platform - OS"),
        release_year=extract_year_from_release(get("Launch - Announced")),
        screen_size=get("Display - Size"),
        screen_resolution=get("Display - Resolution"),
        battery_capacity=get("Battery - Type"),
        main_camera=get(
            "Main Camera - Single",
            "Main Camera - Dual",
            "Main Camera - Triple",
            "Main Camera - Quad"
        ),
        selfie_camera=get("Selfie camera - Single", "Selfie camera - Dual"),
        chipset=get("Platform - Chipset"),
        cpu=get("Platform - CPU"),
        gpu=get("Platform - GPU"),
        network=get("Network - Technology"),
        network_bands=get("Network - 2G bands"),
        sim=get("Network - SIM"),
        weight=get("Body - Weight"),
        dimensions=get("Body - Dimensions"),
        usb=get("Comms - USB"),
        sensors=get("Features - Sensors"),
        price=get("Misc - Price")
    )


    if "Memory - Internal" in specs:
        storage_list, ram_list = parse_ram_and_storage(specs["Memory - Internal"])
        mobile.storage = ", ".join(storage_list)
        mobile.ram = ", ".join(ram_list)

    return mobile



def scrape_models_details(model_detail_urls: List[str]) -> List[NewMobile]:
    # Get existing models from DB
    try:
        stored_names = set()
        for entry in model_names_collection.find({}, {"name": 1, "_id": 0}):
            stored_names.add(entry["name"].lower())
    except Exception as e:
        print(f"Error fetching existing models: {e}")
        stored_names = set()


    for url in model_detail_urls:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, "html.parser")

            title = soup.select_one("h1.specs-phone-name-title")

            full_title = title.text.strip() if title else ""
            parts = full_title.split(" ", 1)

            brand = parts[0]
            model = parts[1] if len(parts) > 1 else None

            brand_model = f"{brand} {model}".strip().lower()

            if brand_model in stored_names:
                print(brand_model)
                print(f"Skipping already saved model: {brand_model}")
                continue

            specs = {}
            specs_list = soup.find("div", id="specs-list")
            current_section = None

            if specs_list:
                for row in specs_list.select("tr"):
                    th = row.find("th", {"scope": "row"})

                    if th:
                        current_section = th.text.strip()

                    key_td = row.find("td", class_="ttl")
                    value_td = row.find("td", class_="nfo")
                    
                    if key_td and value_td and current_section:
                        key = key_td.text.strip()
                        value = value_td.text.strip()
                        specs[f"{current_section} - {key}"] = value


            phone = convert_specs_to_mobile(specs)
            phone.brand = brand
            phone.model = model

            print(phone.model)

            save_to_db(phone)

            time.sleep(3)

        except requests.exceptions.RequestException as e:
            print(f"Network error scraping {url}: {e}")

        except Exception as e:
            print(f"Error scraping {url}: {e}")


def save_to_db(mobile: NewMobile):
    try:
        mobile_dict = mobile.model_dump()

        mobiles_collection.insert_one(mobile_dict)

        brand_model = f"{mobile.brand} {mobile.model}".strip()
        model_names_collection.insert_one({"name": brand_model})

    except Exception as e:
        print(f"Error saving {mobile.brand} {mobile.model}: {e}")


# import pandas as pd

# # Load the CSV file
# df = pd.read_csv("google.csv")

# # Get the first column as a list of strings
# urls = df.iloc[:, 0].dropna().astype(str).tolist()

#scrape_models_details(urls)