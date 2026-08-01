from captchasele import capture_captcha
from captchatess import read_captcha

from selenium.webdriver.common.by import By
import time

# -------------------------
# Open website & capture captcha
# -------------------------

driver, image_path = capture_captcha()

# -------------------------
# OCR
# -------------------------

captcha_text = read_captcha(image_path)

print(f"\nDetected CAPTCHA : {captcha_text}")

# -------------------------
# Fill captcha textbox
# -------------------------

captcha_box = driver.find_element(By.ID, "captchaInput")

captcha_box.clear()

captcha_box.send_keys(captcha_text)

print("Captcha entered successfully!")

# Keep browser open for checking
time.sleep(30)

driver.quit()