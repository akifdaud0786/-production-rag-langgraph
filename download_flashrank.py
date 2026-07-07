import os
import zipfile
from pathlib import Path
import requests

MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "flashrank_cache"
MODEL_DIR = CACHE_DIR / MODEL_NAME
ZIP_PATH = CACHE_DIR / f"{MODEL_NAME}.zip"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_DIR.exists():
    print(f"Downloading FlashRank model {MODEL_NAME}...")
    url = f"https://huggingface.co/prithivida/flashrank/resolve/main/{MODEL_NAME}.zip"
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting zip...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(CACHE_DIR)
        print("Extraction complete. Model is ready!")
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
    except Exception as e:
        print(f"Failed to pre-download model: {e}")
else:
    print(f"Model {MODEL_NAME} already exists locally.")
