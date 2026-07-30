import os
import time
import fitz

from openai import OpenAI

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

# ==========================================================
# Chrome Configuration
# ==========================================================

options = webdriver.ChromeOptions()

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

def wait_for_download(folder, timeout=60):
    """
    Wait until Chrome finishes downloading.
    Returns the newest PDF.
    """

    start = time.time()

    while time.time() - start < timeout:

        files = os.listdir(folder)

        # Chrome still downloading
        if any(f.endswith(".crdownload") for f in files):
            time.sleep(1)
            continue

        pdfs = [
            os.path.join(folder, f)
            for f in files
            if f.lower().endswith(".pdf")
        ]

        if pdfs:
            newest = max(pdfs, key=os.path.getmtime)
            return newest

        time.sleep(1)

    return None


def extract_text(pdf_path):

    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def summarize(text):

    if len(text) > 100000:
        text = text[:100000]

    prompt = f"""
You are an expert legal assistant.

Read the following court order.

Return a concise summary containing:

1. Court
2. Case Number
3. Parties
4. Judge
5. Main Issue
6. Decision
7. Important Directions
8. Next Hearing Date
9. 5 Bullet Point Summary

Court Order:

{text}
"""

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


def save_summary(pdf_path, summary):

    txt_path = pdf_path.replace(".pdf", "_summary.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print("Summary saved:", txt_path)


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

    print(f"\nFound {len(pdf_links)} PDFs\n")

    for pdf in pdf_links:

        print("=" * 70)
        print(pdf)

        # Remember current PDFs
        existing = set(os.listdir(DOWNLOAD_DIR))

        # Trigger download
        driver.get(pdf)

        downloaded_pdf = None

        start = time.time()

        while time.time() - start < 60:

            current = set(os.listdir(DOWNLOAD_DIR))

            new_files = current - existing

            pdfs = [
                f for f in new_files
                if f.lower().endswith(".pdf")
            ]

            downloading = any(
                f.endswith(".crdownload")
                for f in current
            )

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

        text = extract_text(downloaded_pdf)

        if len(text.strip()) < 100:

            print("Very little text found.")
            continue

        print("Extracted", len(text), "characters")

        summary = summarize(text)

        save_summary(downloaded_pdf, summary)

        print(summary[:300])

    print("\nFinished.")

finally:

    driver.quit()