import sys
from bs4 import BeautifulSoup

# --- New Imports for Browser Automation ---
# To run this script, you'll need to install Selenium and webdriver-manager:
# pip install selenium webdriver-manager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def get_weather_data_dynamically(url):
    """
    Fetches the dynamic content of a given URL using Selenium to control a
    headless Chrome browser. It waits for the JavaScript on the page to
    load the data before returning the HTML.

    Args:
        url (str): The URL of the wunderground.com dashboard to fetch.

    Returns:
        str: The full HTML of the page after dynamic content has loaded,
             otherwise None.
    """
    # --- Selenium WebDriver Setup ---
    # Configure Chrome to run in headless mode (no visible browser window)
    # and set the same User-Agent as before.
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36")

    driver = None  # Initialize driver to None
    try:
        # Use webdriver-manager to automatically download and manage the
        # correct version of chromedriver for your installed Chrome browser.
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        # Navigate to the page
        driver.get(url)
        #print("Waiting for dynamic content to load...")

        # --- Wait for the key element to appear ---
        # This is the crucial step. We wait up to 30 seconds for the element
        # containing the temperature to appear. The presence of a `span` with
        # class `wu-value` inside the `div.main-temp` indicates the data has loaded.
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.main-temp span.wu-value"))
        )
        
        #print(f"Successfully fetched dynamic content from: {url}")
        
        # Once the element is present, return the full page source
        return driver.page_source

    except Exception as e:
        print(f"An error occurred while fetching the page with Selenium: {e}", file=sys.stderr)
        return None
    finally:
        # Ensure the browser instance is always closed, even if errors occur.
        if driver:
            driver.quit()

def extract_main_temp(html):
    """
    Parses the HTML content to find and extract the main temperature.
    This function remains the same, as it processes the final HTML.

    Args:
        html (str): The HTML content of the weather page.

    Returns:
        str: The main temperature as a string, or None if not found.
    """
    if not html:
        return None
        
    soup = BeautifulSoup(html, 'html.parser')

    # Find the specific container div for the temperature display first.
    conditions_div = soup.find('div', class_='conditions-temp')
  
    if conditions_div:
        # Now, within that specific container, find the div that contains the main temp.
        main_temp_div = conditions_div.find('div', class_='main-temp')
        if main_temp_div:
            # Within the 'main_temp_div', find the span that contains the value.
            temp_span = main_temp_div.find('span', class_='wu-value')
            if temp_span:
                return temp_span.get_text(strip=True)

    # Return None if the temperature couldn't be found in the expected structure.
    print("Could not find the temperature element using the specified path in the final HTML.", file=sys.stderr)
    return None

# --- Main execution block ---
if __name__ == "__main__":
    # The target URL for the personal weather station dashboard.
    target_url = "https://www.wunderground.com/dashboard/pws/KCAAUBER79"

    # Call the new function to get the weather data using a real browser.
    html_content = get_weather_data_dynamically(target_url)

    if html_content:
        # If HTML was fetched successfully, pass it to the extraction function.
        main_temp = extract_main_temp(html_content)

        if main_temp:
            # If a temperature was found, print it.
            #print("\n--- Extracted Temperature ---")
            print(f"Temperature: {main_temp}°F")
            #print("-----------------------------")
        else:
            # Inform the user if the temperature could not be found.
            print("\nCould not find the main temperature in the page's dynamic content.", file=sys.stderr)
