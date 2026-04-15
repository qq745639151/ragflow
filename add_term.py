import requests
import json

# API endpoint
url = "http://222.90.211.46:20080/api/v1/dictionary/add_term"

# API key
api_key = "ragflow-7VR4ItrAzNQUIljrZzR6mGGwQERNlHgAqtY3oQDXdNQ"

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Term to add
data = {
    "term": "线路串抗",
    "frequency": 3,
    "pos": "n"
}

# Send request
try:
    response = requests.post(url, headers=headers, json=data)
    print("Response status code:", response.status_code)
    print("Response content:", response.text)
except Exception as e:
    print("Error:", str(e))
