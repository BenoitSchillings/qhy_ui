import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class TemperatureScraper:
    """
    Handles fetching the ambient temperature from a Weather Underground PWS page.
    """
    def __init__(self, url):
        self.url = url
        self.driver = None

    def _setup_driver(self):
        """Initializes the headless Chrome browser."""
        print("Initializing browser...")
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36")
        
        try:
            print("Downloading/verifying browser driver... (this may take a moment on first run)")
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print(f"Navigating to {self.url}...")
            self.driver.get(self.url)
        except Exception as e:
            print(f"Error setting up the browser: {e}", file=sys.stderr)
            self.shutdown()
            raise

    def get_current_temperature_fahrenheit(self):
        """
        Fetches the current temperature from the page.

        Returns:
            float: The temperature in Fahrenheit, or None if an error occurs.
        """
        if not self.driver:
            self._setup_driver()

        try:
            print("Fetching updated temperature...")
            self.driver.refresh()
            
            # Wait for the key element to appear, ensuring dynamic content has loaded.
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.main-temp span.wu-value"))
            )
            
            html_content = self.driver.page_source
            temp_str = self._extract_temp_from_html(html_content)
            
            if temp_str:
                return float(temp_str)
            return None

        except Exception as e:
            print(f"An error occurred while fetching temperature: {e}", file=sys.stderr)
            return None

    def _extract_temp_from_html(self, html):
        """Parses HTML to find the temperature value."""
        if not html:
            return None
            
        soup = BeautifulSoup(html, 'html.parser')
        conditions_div = soup.find('div', class_='conditions-temp')
        if conditions_div:
            main_temp_div = conditions_div.find('div', class_='main-temp')
            if main_temp_div:
                temp_span = main_temp_div.find('span', class_='wu-value')
                if temp_span:
                    return temp_span.get_text(strip=True)
        
        print("Could not find temperature element in HTML.", file=sys.stderr)
        return None

    def shutdown(self):
        """Closes the browser instance."""
        if self.driver:
            print("Closing browser.")
            self.driver.quit()
            self.driver = None
