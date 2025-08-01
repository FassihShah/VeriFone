import requests
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

load_dotenv()

IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
BING_END_POINT = os.getenv("BING_END_POINT")


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ModelBrandOutput(BaseModel):
    brand_name: str
    model_name: str


BRANDS = [
    "Samsung", "Apple", "Xiaomi", "Oppo", "Vivo", "Realme", "Infinix",
    "OnePlus", "Huawei", "Tecno", "Nokia", "Sony", "LG", "Google",
    "Motorola", "Lenovo", "Asus", "Honor", "Vgo Tel", "Itel"
]


chain = llm.with_structured_output(ModelBrandOutput)

def extract_model_brand(raw_results):

    if not raw_results:
        raise ValueError("Missing search_results")

    # Extract useful text only
    results = []
    for item in raw_results.get("data", []):
        results.append({
            "title": item.get("title", ""),
            "image_url": item.get("image_url", "")
        })


    text_snippets = "\n".join(f"- {item['title']}" for item in results if item['title'])

    prompt = f"""
You are an expert in identifying smartphone models and brands.

From the text below, extract:
- The **brand name** (must be exactly one of: {', '.join(BRANDS)})
- The **most complete and accurate model name** (e.g., "Pixel 6a")

Instructions:
- The brand name and model name should be returned separately.
- The model name **does not need to include the brand name** (e.g., for "Galaxy S21", brand = Samsung, model = Galaxy S21).
- Only use titles that clearly refer to a smartphone.
- Ignore irrelevant content like updates, opinions, or general tech news.
- Do **not** make up or guess the model name. Only extract names **that are clearly mentioned** in the titles or snippets.
- If multiple model names are mentioned, choose the **most frequently occurring one**.
- Be precise — prefer full model names over partial or generic ones.

Text snippets:
{text_snippets}
"""


    result = chain.invoke(prompt)

    brand_name = result.brand_name
    model_name = result.model_name

    return brand_name, model_name




def search_by_image(image_path: str):

    print(f"🔍 Processing image: {image_path}")
    os_path = image_path.replace('/', os.sep)

    if not os.path.exists(os_path) or not os.path.isfile(os_path):
        raise FileNotFoundError(f"Image file not found or is not a file: {os_path}")

    file_size = os.path.getsize(os_path)
    if file_size == 0:
        raise ValueError(f"Image file is empty: {os_path}")

    if not IMGBB_API_KEY:
        raise ValueError("IMGBB_API_KEY not found in environment variables")

    with open(os_path, "rb") as file:
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_API_KEY},
            files={"image": ("image.jpg", file, "image/jpeg")},
            timeout=60
        )

    print(f"ImgBB response status: {response.status_code}")
    if response.status_code != 200:
        raise RuntimeError(f"ImgBB API error: {response.status_code} - {response.text}")

    imgbb_data = response.json()
    uploaded_url = imgbb_data.get("data", {}).get("url")
    if not uploaded_url:
        raise ValueError(f"ImgBB response missing URL: {imgbb_data}")

    print(f"✅ Uploaded to ImgBB: {uploaded_url}")

    if not RAPID_API_KEY:
        raise ValueError("RAPID_API_KEY not found in environment variables")

    print("🔎 Starting Bing Visual Search...")
    search_response = requests.get(
        BING_END_POINT,
        headers={
            "X-RapidAPI-Host": "bing-image-search5.p.rapidapi.com",
            "X-RapidAPI-Key": RAPID_API_KEY
        },
        params={"query_url": uploaded_url},
        timeout=60
    )

    print(f"Bing response status: {search_response.status_code}")
    if search_response.status_code != 200:
        raise RuntimeError(f"Bing Visual Search failed: {search_response.status_code} - {search_response.text}")

    search_results = search_response.json()
    print("✅ Visual search completed successfully")

    brand, model = extract_model_brand(search_results)

    return brand, model



# brand, name = search_by_image("services/Samsung-Galaxy-S23-Ultra.jpg")

# print(brand,name)
