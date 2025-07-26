import requests
import time

while(True):
    url = "https://sroftp2.com/SRODATA/AlcorSM/Last_Seeing_Data.txt"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        content = response.text
        print(content)
    else:
        print(f"Failed to download file. Error code: {response.status_code}")

    time.sleep(200)