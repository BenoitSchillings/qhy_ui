import sys
import time  # Import the time module
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

def extract_main_temp(html):
    """
    Parses the HTML content to find and extract the main temperature.

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
    print("Could not find the temperature element in the page HTML.", file=sys.stderr)
    return None

# --- Main execution block ---
if __name__ == "__main__":
    # The target URL for the personal weather station dashboard.
    target_url = "https://www.wunderground.com/dashboard/pws/KCAAUBER79"

    # --- Selenium WebDriver Setup ---
    # Configure Chrome to run in headless mode (no visible browser window)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36")

    driver = None  # Initialize driver to None
    try:
        # Use webdriver-manager to automatically download and manage chromedriver
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"Opening {target_url}...")
        # Navigate to the page once at the beginning
        driver.get(target_url)

        while True:
            try:
                #print("Fetching updated temperature...")
                
                # --- Wait for the key element to appear ---
                # This ensures the dynamic content has loaded after the initial page load or a refresh.
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.main-temp span.wu-value"))
                )
                
                # Get the page source after data has loaded
                html_content = driver.page_source
                main_temp = extract_main_temp(html_content)

                if main_temp:
                    print(f"{main_temp}°F")
                else:
                    print("Could not extract temperature.", file=sys.stderr)

                # Wait for 5 seconds before the next refresh
                
                time.sleep(60)
                
                # Refresh the page to get new data
                driver.refresh()

            except KeyboardInterrupt:
                # Allow user to exit the loop with Ctrl+C
                print("\nExiting program.")
                break
            except Exception as e:
                print(f"An error occurred during the loop: {e}", file=sys.stderr)
                # Optional: break the loop on other errors, or just wait and retry
                time.sleep(10) # Wait a bit longer after an error before retrying
                driver.refresh() # Try to refresh to recover

    except Exception as e:
        print(f"A critical error occurred: {e}", file=sys.stderr)
    finally:
        # Ensure the browser instance is always closed, even if errors occur.
        if driver:
            print("Closing browser.")
            driver.quit()
