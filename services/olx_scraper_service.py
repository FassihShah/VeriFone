from bs4 import BeautifulSoup
import requests
from time import sleep
import random
import pandas as pd
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId
import os
import time
from models import UsedMobile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# Constants
BASE_URL = "https://www.olx.com.pk/mobile-phones_c1411"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Mongo Setup
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
DB_NAME = "MobileDB"
COLLECTION_NAME = "used_mobiles"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# LLM Setup
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

model_verification_prompt = ChatPromptTemplate.from_template("""
You are an assistant that checks whether a used mobile listing **matches exactly** the given brand and model (including full variant name).

You are given:
- Title: {title}
- Description: {description}
- Expected Brand: {brand}
- Expected Model: {model}

Return `"true"` only if both brand and model in the title or description **exactly match** the expected values. This includes matching full model names such as "Pixel 6A", "A52s", "iPhone 13 Pro Max".

Reject listings that mention:
- A different model (e.g., "Pixel 8" vs "Pixel 6A")
- A different variant (e.g., "A52s" vs "A52")
- Only partial matches (e.g., "Pixel" or "Samsung" without matching the full model)

### Examples:
- ✅ "Google Pixel 6A in mint condition" → "true" if expected model is "Pixel 6A"
- ❌ "Pixel 6" → "false" if expected model is "Pixel 6A"
- ❌ "Pixel 8 with box" → "false" if expected model is "Pixel 6A"
- ✅ "Samsung Galaxy A52s PTA approved" → "true" if expected model is "A52s"

Return only **"true"** or **"false"**, nothing else.
""")



main_extraction_prompt = ChatPromptTemplate.from_template("""
You are a smart assistant that extracts structured information from used mobile listings. You are given the following raw input fields scraped from OLX:

- Title: {title}
- Description: {description}
- Brand: {brand}
- Model: {model}
- Overall User Condition Note (free text): {condition}
- Price: {price}
- Location: {location}

Using this information, return a JSON object with the following fields:

- brand  
- model  
- ram  
- storage  
- condition (Rate out of 10 based on user's tone and overall language, e.g. "10/10", "good condition", etc. — ignore technical issues like fingerprint problems, panel cracks, etc.)  
- pta_approved  
- is_panel_changed  
- screen_crack  
- panel_dot  
- panel_line  
- panel_shade
- camera_lens_ok  
- fingerprint_ok  
- with_box  
- with_charger  
- price  
- city  
- listing_source (always "OLX")  
- images (list of URLs)  
- post_date

### Assumptions:
- If the listing **does not explicitly mention any problem** related to display, fingerprint, screen, or accessories, **assume everything is OK (i.e., set those fields accordingly):**
    - is_panel_changed → false  
    - screen_crack → false  
    - panel_dot → false  
    - panel_line → false
    - panel_shade → false
    - camera_lens_ok → true  
    - fingerprint_ok → true  
    - with_box → false  
    - with_charger → false

- If **PTA approval is not mentioned explicitly**, assume the device is **PTA approved**. However, set `pta_approved` to `false` if any of the following terms appear:  
    - "non PTA"  
    - "PTA not approved"  
    - "SIM lock"  
    - "JV phone"  

- Do not leave these assumption-based fields as null unless something directly contradicts the rule.

- Return `null` for any **other field** that cannot be confidently extracted from the input.

- Use only what's present in the input; **do not hallucinate or guess beyond the assumption rules.**

**Guidance for 'condition' field (out of 10):**

Lower the score (3–7) if the tone suggests:  
- "scratches", "scratchy", "visible marks"  
- "rough use", "daily used", "used for 2+ years"  
- "broken back", "dented", "chipped", "damaged"  
- "repaired"

Raise the score (8–10) if the tone includes:  
- "scratchless", "like new", "excellent condition"  
- "carefully used", "single hand used", "used with care"  
- "10/10", "almost new", "fresh condition", "mint"

Focus only on subjective tone and handling for the `condition` score. Ignore technical issues — they're handled separately.

Return the result strictly as a **JSON object**.
""")

model_verification_chain = model_verification_prompt | llm | StrOutputParser()
data_extraction_chain = main_extraction_prompt | llm.with_structured_output(UsedMobile)



last_gemini_call = 0

def rate_limit_pause():
    global last_gemini_call
    now = time.time()
    elapsed = now - last_gemini_call
    min_interval = 6

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    last_gemini_call = time.time()



def save_to_db(mobile: UsedMobile):
    collection = db[COLLECTION_NAME]
    now = datetime.now(timezone.utc)
    data = mobile.model_dump()
    data["extraction_date"] = now
    data["_id"] = ObjectId()

    try:
        collection.insert_one(data)
        print("✅ Data saved successfully.")
        return True
    except Exception as e:
        print("❌ MongoDB insert error:", e)
        return False


def extract_data(data: dict, model, brand):
    try:
        rate_limit_pause()
        result = model_verification_chain.invoke({
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "brand": brand,
            "model": model
        })


        if result.strip().lower() != "true":
            print("❌ Skipped: model/brand mismatch :", data.get("title",""))
            return False
        
        
        rate_limit_pause()
        mobile: UsedMobile = data_extraction_chain.invoke({
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "brand": brand,
            "model": model,
            "condition": data.get("condition", ""),
            "price": data.get("price", ""),
            "location": data.get("location", "")
        })


        mobile.post_date = data.get("post_date", "")
        mobile.images = data.get("images", "")

        success = save_to_db(mobile)
        print(f"✅ Extracted: {mobile.model}")
        if success:
            return True

    except Exception as e:
        print("❌ Extraction failed:", e)
        return False


def get_ads_from_page(page_num, model_query, brand):

    if brand.lower() not in model_query.lower():
        full_query = f"{brand} {model_query}"
    else:
        full_query = model_query

    encoded_query = quote_plus(full_query)
    url = f"{BASE_URL}/q-{encoded_query}/?page={page_num}"
    print(f"Scraping: {url}")

    scraper = requests.Session()
    scraper.headers.update(HEADERS)
    res = scraper.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    ads = soup.select("li[aria-label='Listing']")
    listings = []

    for ad in ads:
        try:
            title_tag = ad.select_one("h2._1093b649")
            price_tag = ad.select_one("div[aria-label='Price'] span")
            location_tag = ad.select_one("span.f047db22")
            link_tag = ad.find("a", href=True)

            if not all([title_tag, price_tag, location_tag, link_tag]):
                continue

            title = title_tag.text.strip()
            price = price_tag.text.strip()
            location = location_tag.text.strip()
            link = "https://www.olx.com.pk" + link_tag["href"]

            ad_res = scraper.get(link)
            ad_soup = BeautifulSoup(ad_res.text, "html.parser")

            desc_tag = ad_soup.select_one("div[aria-label='Description'] div._7a99ad24 span")
            description = desc_tag.text.strip() if desc_tag else ""

            details = {}
            detail_tags = ad_soup.select("div[aria-label='Details'] div._0272c9dc.cd594ce1")
            for detail in detail_tags:
                spans = detail.find_all("span")
                if len(spans) == 2:
                    details[spans[0].text.strip()] = spans[1].text.strip()

            image_tags = ad_soup.select("div.image-gallery-slide img")
            image_urls = [img['src'] for img in image_tags if img.get('src')]

            data = {
                "title": title,
                "price": price,
                "location": location,
                "link": link,
                "description": description,
                "brand": details.get("Brand", ""),
                "model": details.get("Model", ""),
                "condition": details.get("Condition", ""),
                "images": ", ".join(image_urls)
            }

            success = extract_data(data, model_query, brand)
            if success:
                listings.append(data)

        except Exception as e:
            print("Skipping ad due to error:", e)
            continue

    return listings


def scrape_used_data(model: str, brand: str):
    print(f"Collecting data for model: {model}")
    all_listings = []
    page_num = 1

    try:
        while True:
            listings = get_ads_from_page(page_num, model, brand)
            if not listings:
                print(f"No listings found on page {page_num}. Stopping.")
                break

            all_listings.extend(listings)
            if len(all_listings) >= 100:
                print("Collected enough listings (100+). Stopping.")
                break

            page_num += 1
            sleep(random.uniform(3, 6))

    except Exception as e:
        print(f"❌ Error while scraping data: {e}")

    df = pd.DataFrame(all_listings)
    file_name = f"{brand}_{model}.csv".replace(" ", "_").lower()
    df.to_csv(file_name, index=False, encoding="utf-8-sig")

    print(f"✅ Saved {len(df)} listings to {file_name}")
    print(f"✅ Collected {len(all_listings)} listings.")



#scrape_used_data("Galaxy A32", "Samsung")
