import os
import time
import fitz

from openai import OpenAI, RateLimitError

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ==========================================================
# Configuration
# ==========================================================

BASE_URL = "https://aftlko.up.nic.in/"

DOWNLOAD_DIR = os.path.abspath("downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

client = OpenAI(
    api_key="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"
)

# Number of PDFs to summarize using OpenAI
MAX_SUMMARIES = 3

# ==========================================================
# Chrome Configuration
# ==========================================================

options = webdriver.ChromeOptions()

# Uncomment to run headless
# options.add_argument("--headless=new")

prefs = {
    "download.default_directory": DOWNLOAD_DIR,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
}

options.add_experimental_option("prefs", prefs)

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

driver.maximize_window()

# ==========================================================
# Helper Functions
# ==========================================================

def extract_text(pdf_path):
    """Extract text from PDF"""

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def save_text(pdf_path, text):
    """Save extracted text"""

    txt_path = pdf_path.replace(".pdf", ".txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("Text saved:", txt_path)


def summarize(text):
    """Summarize using OpenAI"""

    if len(text) > 100000:
        text = text[:100000]

    prompt = f"""
You are an expert legal assistant.

Read the following court order and return:

1. Court
2. Case Number
3. Parties
4. Judge
5. Main Issue
6. Decision
7. Important Directions
8. Next Hearing Date
9. Five Bullet Point Summary

Court Order:

{text}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize legal court orders."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except RateLimitError:

        print("\nOpenAI quota exceeded.")
        return None

    except Exception as e:

        print("OpenAI Error:", e)
        return None


def save_summary(pdf_path, summary):
    """Save summary"""

    summary_path = pdf_path.replace(".pdf", "_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("Summary saved:", summary_path)


# ==========================================================
# Main
# ==========================================================

try:

    print("Opening Website...")
    driver.get(BASE_URL)

    time.sleep(3)

    print("Opening Cause List...")

    driver.find_element(
        By.XPATH,
        "//a[contains(@href,'CAUSE LIST.html')]"
    ).click()

    time.sleep(3)

    links = driver.find_elements(By.TAG_NAME, "a")

    pdf_links = []

    for link in links:

        href = link.get_attribute("href")

        if href and href.lower().endswith(".pdf"):
            pdf_links.append(href)

    print(f"\nFound {len(pdf_links)} PDF links\n")

    summaries_done = 0

    for pdf in pdf_links:

        print("=" * 80)
        print("Downloading:", pdf)

        existing = set(os.listdir(DOWNLOAD_DIR))

        driver.get(pdf)

        downloaded_pdf = None

        start = time.time()

        while time.time() - start < 60:

            current = set(os.listdir(DOWNLOAD_DIR))

            new_files = current - existing

            downloading = any(
                f.endswith(".crdownload")
                for f in current
            )

            pdfs = [
                f for f in new_files
                if f.lower().endswith(".pdf")
            ]

            if pdfs and not downloading:

                downloaded_pdf = os.path.join(
                    DOWNLOAD_DIR,
                    pdfs[0]
                )

                break

            time.sleep(1)

        if downloaded_pdf is None:

            print("Download failed.")
            continue

        print("Downloaded:", downloaded_pdf)

        # --------------------------------------------------
        # Extract Text
        # --------------------------------------------------

        text = extract_text(downloaded_pdf)

        print(f"Extracted {len(text)} characters")

        save_text(downloaded_pdf, text)

        # --------------------------------------------------
        # Skip scanned PDFs
        # --------------------------------------------------

        if len(text.strip()) < 100:

            print("Very little text found. Skipping summary.\n")
            continue

        # --------------------------------------------------
        # Stop after N summaries
        # --------------------------------------------------

        if summaries_done >= MAX_SUMMARIES:

            print(f"\nReached limit of {MAX_SUMMARIES} summaries.")
            break

        print("Generating summary...")

        summary = summarize(text)

        if summary is None:
            print("Stopping further processing.")
            break

        save_summary(downloaded_pdf, summary)

        summaries_done += 1

        print(f"Completed {summaries_done}/{MAX_SUMMARIES} summaries\n")

        print("=" * 80)
        print(summary[:500])
        print("=" * 80)

    print("\nFinished.")

finally:

    driver.quit()