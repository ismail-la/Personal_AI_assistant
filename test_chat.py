import requests

resp = requests.post(
    "http://127.0.0.1:8000/chat/",
    json={"prompt": "Summarize the benefits of AI."}
)
print(resp.json()["response"])
