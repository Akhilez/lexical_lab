import json
import requests

url = "https://www.csci555competition.online/score"

payload = json.dumps(predictions.tolist())
headers = {"Content-Type": "application/json"}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
