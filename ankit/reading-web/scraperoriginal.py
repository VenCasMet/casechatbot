import time
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

AFT_REGIONAL_BENCHES_URL = "https://aft.gov.in/reg-benches"
CAUSE_LIST_URL = "https://aftlko.up.nic.in/CAUSE%20LIST.html"
DOWNLOAD_DIR = Path("downloads2").resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Windows only: point this at the "bin" folder inside your extracted Poppler
# download, e.g. r"C:\poppler\Library\bin" (must contain pdftoppm.exe / pdfinfo.exe).
# On Mac/Linux where Poppler is installed via brew/apt, leave this as None.
POPPLER_PATH = r"C:\poppler\Library\bin"

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def create_driver(download_dir: Path) -> webdriver.Chrome:
    options = webdriver.ChromeOptions()

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    }

    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.maximize_window()
    return driver


def extract_text(pdf_path: Path) -> str:
    """
    These cause-list PDFs are scanned images (no embedded text layer),
    so a plain text extractor (pdfplumber/pypdf) always returns "".
    We OCR each page instead using pytesseract.
    """
    text = ""
    try:
        # 300 dpi is a good balance of OCR accuracy vs speed for this kind
        # of document (tables with small print). Bump to 400 if accuracy
        # is still rough on some scans.
        images = convert_from_path(
            str(pdf_path),
            dpi=300,
            poppler_path=POPPLER_PATH if POPPLER_PATH else None,
        )
        for i, image in enumerate(images, 1):
            # --psm 6: assume a single uniform block of text (works well
            # for table-like layouts like this cause list).
            page_text = pytesseract.image_to_string(image, config="--psm 6")
            text += f"--- Page {i} ---\n{page_text}\n"
    except Exception as e:
        print(f"Warning: Could not OCR {pdf_path}: {e}")
    return text


def save_text(pdf_path: Path, text: str) -> Path:
    txt_path = pdf_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    print("Text saved:", txt_path)
    return txt_path


def open_regional_benches(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    print("Opening Regional Benches...")
    driver.get(AFT_REGIONAL_BENCHES_URL)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))


def find_lucknow_url(driver: webdriver.Chrome) -> str:
    for link in driver.find_elements(By.TAG_NAME, "a"):
        href = link.get_attribute("href")
        if href and "aftlko.up.nic.in" in href.lower():
            return href
    raise RuntimeError("Lucknow Bench URL not found on AFT Regional Benches page.")


def open_cause_list(driver: webdriver.Chrome, wait: WebDriverWait) -> None:
    print("Opening Cause List...")
    driver.get(CAUSE_LIST_URL)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "a")))


def collect_pdf_urls(driver: webdriver.Chrome) -> list[str]:
    pdf_urls = []
    for link in driver.find_elements(By.TAG_NAME, "a"):
        href = link.get_attribute("href")
        if href and href.lower().endswith(".pdf"):
            pdf_urls.append(href)
    return list(dict.fromkeys(pdf_urls))


def wait_for_download(download_dir: Path, existing_paths: set[Path], timeout: int = 10) -> Path:
    start = time.time()
    while time.time() - start < timeout:
        current_paths = set(download_dir.iterdir())
        new_paths = current_paths - existing_paths
        downloading = any(path.suffix == ".crdownload" for path in current_paths)
        pdf_paths = [path for path in new_paths if path.suffix.lower() == ".pdf"]
        if pdf_paths and not downloading:
            return pdf_paths[0]
        time.sleep(1)
    raise RuntimeError("Download failed or timed out.")


def download_pdf(driver: webdriver.Chrome, pdf_url: str, download_dir: Path) -> Path:
    print("Downloading:", pdf_url)
    existing_paths = set(download_dir.iterdir())
    driver.get(pdf_url)
    downloaded_pdf = wait_for_download(download_dir, existing_paths)
    print("Downloaded:", downloaded_pdf)
    return downloaded_pdf


def main() -> None:
    driver = create_driver(DOWNLOAD_DIR)
    wait = WebDriverWait(driver, 20)

    try:
        open_regional_benches(driver, wait)
        lucknow_url = find_lucknow_url(driver)
        print("Lucknow:", lucknow_url)

        driver.get(lucknow_url)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        open_cause_list(driver, wait)

        pdf_urls = collect_pdf_urls(driver)
        print(f"\nFound {len(pdf_urls)} PDFs\n")
        if not pdf_urls:
            raise RuntimeError("No PDF links found on the Cause List page.")

        successful_count = 0

        for idx, pdf_url in enumerate(pdf_urls, 1):
            try:
                print(f"[{idx}/{len(pdf_urls)}] Downloading: {pdf_url}")
                downloaded_pdf = download_pdf(driver, pdf_url, DOWNLOAD_DIR)
                text = extract_text(downloaded_pdf)
                print(f"Extracted {len(text)} characters (OCR)")
                save_text(downloaded_pdf, text)
                successful_count += 1
            except Exception as exc:
                print(f"Warning: failed to process {pdf_url}. Skipping to next. ({exc})\n")

        if successful_count == 0:
            raise RuntimeError("All PDF downloads failed; no PDF could be processed.")

        print(f"\n========== SUMMARY ==========")
        print(f"Successfully processed: {successful_count}/{len(pdf_urls)} PDFs")
        print(f"============================\n")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()