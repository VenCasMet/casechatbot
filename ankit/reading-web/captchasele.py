from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os


def capture_captcha():

    filename = os.path.join(os.path.dirname(__file__), "captcha.png")

    driver = webdriver.Chrome()

    driver.get("https://delhihighcourt.nic.in/app/get-case-type-status")

    driver.maximize_window()

    wait = WebDriverWait(driver, 20)

    captcha = wait.until(
        EC.visibility_of_element_located((By.ID, "captcha-code"))
    )

    captcha.screenshot(filename)

    print("Screenshot saved.")

    return driver, filename