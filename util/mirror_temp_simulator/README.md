# Mirror Temperature Simulator

This project simulates the thermal evolution of a large telescope mirror by fetching real-time ambient temperature data and applying a physics-based model.

## Setup

1.  **Install Python:** Ensure you have Python 3.6+ installed.
2.  **Install Dependencies:** Install the required Python libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Google Chrome:** This tool uses a headless version of Google Chrome for web scraping. Please ensure you have Google Chrome installed on your system.

## Running the Simulator

To start the application, run the `main.py` script:

```bash
python main.py
```

The simulator will start, fetch the current ambient temperature, and begin displaying the calculated mirror temperature in your terminal. The display will update periodically.
