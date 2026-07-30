"""
captcha_ocr.py

Multi-pipeline OCR preprocessing experiment for distorted CAPTCHA images.

For every image in INPUT_FOLDER, this script:
  1. Generates several independent preprocessing variants (Otsu, adaptive
     threshold, CLAHE, sharpened, blackhat/tophat line-removal, etc.)
  2. Runs Tesseract on each variant via image_to_data
  3. Computes the average per-character confidence for each variant
  4. Picks the variant with the highest confidence as the "best" result
  5. Saves every variant to its own subfolder (for visual comparison)
  6. Prints a summary report and writes a CSV log

Requires: opencv-python, numpy, pytesseract, Tesseract-OCR installed locally.
"""

import os
import csv
import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# Point this at your local tesseract binary (Windows example shown).
# On Linux/Mac, comment this out if tesseract is already on PATH.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_FOLDER = "captcha"
OUTPUT_FOLDER = "output"
LOG_CSV = os.path.join(OUTPUT_FOLDER, "results_log.csv")

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# CAPTCHA text is alphanumeric and mixed-case, so DO NOT whitelist only
# letters - that's exactly what turns '9' into 'O' etc. Keep both cases
# and digits; whitelisting is still useful to block stray symbol noise.
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# Try a couple of PSMs per image - single-line (7/8) works well once the
# image is cleanly segmented, 13 (raw line, no layout analysis) is a good
# fallback for oddly-shaped CAPTCHA text.
PSM_CANDIDATES = [7, 8, 13]
OEM = 1  # LSTM engine

def build_config(psm: int) -> str:
    return (
        f"--oem {OEM} --psm {psm} "
        f"-c tessedit_char_whitelist={WHITELIST} "
        f"-c load_system_dawg=0 -c load_freq_dawg=0"
    )


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class VariantResult:
    name: str
    image: np.ndarray
    text: str = ""
    confidence: float = -1.0
    psm_used: int = -1


# --------------------------------------------------------------------------
# PREPROCESSING BUILDING BLOCKS
# --------------------------------------------------------------------------

def resize(img, scale=4):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def apply_clahe(gray, clip=3.0, tile=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization - boosts local
    contrast, which helps when CAPTCHA characters have uneven shading."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)


def sharpen(gray):
    """Unsharp-mask style sharpening kernel to crisp up character edges
    that get softened by resizing/blurring."""
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def denoise(gray):
    """Non-local means denoising - removes speckle noise while keeping
    edges sharper than a Gaussian/median blur would."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def remove_lines(gray):
    """Attempt to erase thin straight interference lines (a very common
    CAPTCHA distortion) using directional morphological filtering.
    We detect long horizontal/vertical structures, then inpaint them out."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    line_mask = cv2.bitwise_or(horiz_lines, vert_lines)
    # Dilate mask slightly so inpainting fully covers the line width
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    inpainted = cv2.inpaint(gray, line_mask, 3, cv2.INPAINT_TELEA)
    return inpainted


def otsu_threshold(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_threshold(gray, block_size=31, c=10):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def blackhat(gray, ksize=(9, 9)):
    """Highlights dark text on a lighter/noisy background."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)


def tophat(gray, ksize=(9, 9)):
    """Highlights light text on a darker background."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


def morph_clean(binary, open_size=(2, 2), close_size=(3, 3)):
    """Opening removes small speckle noise; closing reconnects character
    strokes that thresholding may have broken."""
    open_kernel = np.ones(open_size, np.uint8)
    close_kernel = np.ones(close_size, np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return closed


def deskew(binary):
    """Estimates and corrects small rotation using the minAreaRect of
    foreground pixels. Skips correction if the detected angle is
    implausibly large (likely noise, not real skew)."""
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 20:
        return binary

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > 15:  # implausible for CAPTCHA text -> don't rotate
        return binary

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def ensure_black_text_on_white(binary):
    """Tesseract prefers black text on white background. If the image is
    mostly foreground (white), invert it."""
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio < 0.5:
        return cv2.bitwise_not(binary)
    return binary


# --------------------------------------------------------------------------
# PIPELINE DEFINITIONS
# --------------------------------------------------------------------------
# Each pipeline takes the original BGR image and returns a final binary
# (or grayscale) image ready for OCR. Keeping them independent means a
# technique that fails on one image doesn't drag down the others.

def pipeline_otsu_basic(img):
    gray = to_gray(resize(img, 5))
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = otsu_threshold(gray)
    return morph_clean(thresh, (2, 2), (2, 2))


def pipeline_adaptive(img):
    gray = to_gray(resize(img, 5))
    gray = denoise(gray)
    thresh = adaptive_threshold(gray, block_size=35, c=11)
    return morph_clean(thresh, (2, 2), (3, 3))


def pipeline_clahe_otsu(img):
    gray = to_gray(resize(img, 5))
    gray = apply_clahe(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = otsu_threshold(gray)
    return morph_clean(thresh, (2, 2), (2, 2))


def pipeline_sharpen_otsu(img):
    gray = to_gray(resize(img, 5))
    gray = sharpen(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = otsu_threshold(gray)
    return morph_clean(thresh, (2, 2), (2, 2))


def pipeline_line_removal(img):
    """Targets the 'background interference lines confuse OCR' failure
    mode directly by inpainting detected line structures first."""
    gray = to_gray(resize(img, 5))
    gray = remove_lines(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = otsu_threshold(gray)
    return morph_clean(thresh, (2, 2), (2, 2))


def pipeline_blackhat(img):
    gray = to_gray(resize(img, 5))
    bh = blackhat(gray, (11, 11))
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    thresh = otsu_threshold(bh)
    return morph_clean(thresh, (2, 2), (2, 2))


def pipeline_color_saturation(img):
    """For CAPTCHAs where the text is a distinct color (e.g. pink/magenta)
    on a plain white/light background. Grayscale intensity alone can be a
    weak signal here, but saturation isolates 'colored' pixels from
    'white/gray' pixels very cleanly regardless of the exact hue."""
    resized = resize(img, 5)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]  # saturation channel: colored text lights up here
    thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cleaned = morph_clean(thresh, (2, 2), (2, 2))
    # Text should be black-on-white for Tesseract
    return ensure_black_text_on_white(cv2.bitwise_not(cleaned))


def pipeline_component_filter(img):
    """Removes crossing/scratch lines and dashed border noise using
    connected-component filtering rather than blind morphology.

    Key idea: in most CAPTCHAs the actual character strokes form large
    connected blobs, while interference lines, dots, and dashed borders
    are made of many small/thin components. Naive opening either leaves
    line fragments behind or erodes thin character strokes along with
    them. This does better by:
      1. Thresholding on saturation (isolates colored text robustly).
      2. Dilating a COPY of the mask so that small disconnected pieces
         of a single character (e.g. the dot on a 'j' or 'i', or a
         stroke segment separated by a crossing line) merge into one
         blob for the purpose of deciding what's "real".
      3. Keeping only components above an area threshold, using the
         dilated version purely to decide membership.
      4. Applying that decision back onto the ORIGINAL (undilated)
         pixels, so character shapes stay crisp instead of fattened.
    """
    resized = resize(img, 5)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    thresh = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Step 2: dilate a working copy to bridge small gaps/fragments
    merge_kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(thresh, merge_kernel, iterations=1)

    # Step 3: keep components above an area threshold (tuned relative to
    # the 5x-upscaled resolution; scales with image size automatically
    # since it's a fraction of total pixel count)
    min_area = max(300, int(0.0015 * resized.shape[0] * resized.shape[1]))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    keep_mask = np.zeros_like(thresh)
    for i in range(1, n):
        if stats[i][cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == i] = 255

    # Step 4: apply the keep-decision back onto the crisp original pixels
    final_mask = cv2.bitwise_and(thresh, keep_mask)
    cleaned = morph_clean(final_mask, (2, 2), (2, 2))
    return ensure_black_text_on_white(cv2.bitwise_not(cleaned))


def pipeline_deskewed_otsu(img):
    gray = to_gray(resize(img, 5))
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = otsu_threshold(gray)
    thresh = ensure_black_text_on_white(cv2.bitwise_not(thresh))
    deskewed = deskew(cv2.bitwise_not(thresh))
    return morph_clean(deskewed, (2, 2), (2, 2))


PIPELINES = {
    "otsu_basic": pipeline_otsu_basic,
    "adaptive_thresh": pipeline_adaptive,
    "clahe_otsu": pipeline_clahe_otsu,
    "sharpen_otsu": pipeline_sharpen_otsu,
    "line_removal": pipeline_line_removal,
    "blackhat": pipeline_blackhat,
    "deskewed_otsu": pipeline_deskewed_otsu,
    "color_saturation": pipeline_color_saturation,
    "component_filter": pipeline_component_filter,
}


# --------------------------------------------------------------------------
# OCR + CONFIDENCE SCORING
# --------------------------------------------------------------------------

def run_ocr(image, psm_candidates=PSM_CANDIDATES):
    """Runs OCR across a few PSMs for one preprocessed image and returns
    the (text, avg_confidence, psm) of the best-scoring PSM attempt."""
    best_text, best_conf, best_psm = "", -1.0, -1

    # Tesseract handles glyphs touching the image border poorly - a
    # generous white margin measurably improves recognition.
    padded = cv2.copyMakeBorder(image, 25, 25, 25, 25,
                                 cv2.BORDER_CONSTANT, value=255)

    for psm in psm_candidates:
        cfg = build_config(psm)
        try:
            data = pytesseract.image_to_data(
                padded, config=cfg, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError:
            continue

        words, confs = [], []
        for word, conf in zip(data["text"], data["conf"]):
            word = word.strip()
            conf = float(conf)
            if word and conf > 0:  # -1 conf = no text detected in that box
                words.append(word)
                confs.append(conf)

        if not confs:
            continue

        avg_conf = sum(confs) / len(confs)
        text = "".join(words)

        if avg_conf > best_conf:
            best_text, best_conf, best_psm = text, avg_conf, psm

    return best_text, best_conf, best_psm


# --------------------------------------------------------------------------
# MAIN DRIVER
# --------------------------------------------------------------------------

def ensure_output_dirs():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for name in PIPELINES:
        os.makedirs(os.path.join(OUTPUT_FOLDER, name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "best"), exist_ok=True)


def pick_best_variant(variants):
    """Selects the best OCR result using confidence AND cross-pipeline
    agreement, not confidence alone.

    Why: Tesseract's own confidence score is not reliable for stylized/
    distorted fonts - a pipeline can be very 'confident' about a wrong
    reading while a correct reading scores lower. If multiple independent
    pipelines land on the same (or a very similar) string, that agreement
    is stronger evidence of correctness than any single confidence value.
    """
    valid = [v for v in variants if v.text]
    if not valid:
        return max(variants, key=lambda v: v.confidence)

    def normalize(t):
        return t.strip().lower()

    from collections import Counter
    votes = Counter(normalize(v.text) for v in valid)

    def score(v):
        agreement = votes[normalize(v.text)]
        # Agreement (how many pipelines produced this exact text) is the
        # primary signal; confidence only breaks ties within that level.
        return (agreement, v.confidence)

    return max(valid, key=score)


def process_image(filename, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[SKIP] Couldn't read {filename}")
        return None

    variants = []
    for name, pipeline_fn in PIPELINES.items():
        try:
            processed = pipeline_fn(img)
        except Exception as e:
            print(f"  [WARN] pipeline '{name}' failed on {filename}: {e}")
            continue

        # Save every variant for visual comparison
        out_path = os.path.join(OUTPUT_FOLDER, name, filename)
        cv2.imwrite(out_path, processed)

        text, conf, psm = run_ocr(processed)
        variants.append(VariantResult(name=name, image=processed,
                                       text=text, confidence=conf, psm_used=psm))

    if not variants:
        print(f"[FAIL] No pipeline produced a result for {filename}")
        return None

    # Pick the variant using confidence + cross-pipeline agreement
    best = pick_best_variant(variants)

    # Save the winning image separately for quick review
    best_path = os.path.join(OUTPUT_FOLDER, "best", filename)
    cv2.imwrite(best_path, best.image)

    return best, variants


def main():
    ensure_output_dirs()

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        return

    filenames = [f for f in sorted(os.listdir(INPUT_FOLDER))
                 if f.lower().endswith(VALID_EXTENSIONS)]

    if not filenames:
        print(f"No images found in '{INPUT_FOLDER}'.")
        return

    rows = []
    for filename in filenames:
        image_path = os.path.join(INPUT_FOLDER, filename)
        result = process_image(filename, image_path)
        if result is None:
            continue
        best, all_variants = result

        print(f"\nImage: {filename}")
        print(f"  Best technique : {best.name} (psm={best.psm_used})")
        print(f"  Avg confidence : {best.confidence:.2f}")
        print(f"  Recognized text: {best.text}")

        rows.append({
            "filename": filename,
            "best_pipeline": best.name,
            "psm": best.psm_used,
            "avg_confidence": round(best.confidence, 2),
            "recognized_text": best.text,
            "all_scores": "; ".join(f"{v.name}={v.confidence:.1f}" for v in all_variants),
        })

    # Write CSV log
    if rows:
        with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nLog written to {LOG_CSV}")


if __name__ == "__main__":
    main()