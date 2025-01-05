from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import random
import time

# Set up the WebDriver (Make sure to replace 'path/to/chromedriver' with the actual path to your chromedriver)
service = Service('path/to/chromedriver')
driver = webdriver.Chrome(service=service)

# URL of the form
url = "http://192.168.72.50:5000/notify"

try:
    # Open the form
    driver.get(url)

    # Wait for the page to load
    time.sleep(2)

    # Input random values for Name
    name_field = driver.find_element(By.ID, "name")
    random_name = f"User{random.randint(1000, 9999)}"
    name_field.send_keys(random_name)

    # Input random values for Phone Number
    phone_field = driver.find_element(By.ID, "phone")
    random_phone = f"{random.randint(1000000000, 9999999999)}"
    phone_field.send_keys(random_phone)

    # Input random values for Email Address
    email_field = driver.find_element(By.ID, "email")
    random_email = f"user{random.randint(1000, 9999)}@example.com"
    email_field.send_keys(random_email)

    # Input random values for Description
    description_field = driver.find_element(By.ID, "description")
    random_description = f"This is a test project description {random.randint(1, 100)}."
    description_field.send_keys(random_description)

    # Select a random star rating (1-5)
    star_rating = random.randint(1, 5)
    stars = driver.find_elements(By.CLASS_NAME, "star")
    actions = ActionChains(driver)
    actions.click(stars[5 - star_rating]).perform()

    # Submit the form
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit']")
    submit_button.click()

    # Wait for the response page to load
    time.sleep(2)

finally:
    # Close the browser
    driver.quit()
