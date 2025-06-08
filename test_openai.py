import httpx
from openai import OpenAI

API_URL = "http://0.0.0.0:8000/v1"
client = OpenAI(api_key="sk-xxx", base_url=API_URL)

# Example: POST to a custom endpoint
response = client.post(
    "/chat/completions",  # or any custom path
    cast_to=httpx.Response,
    body={
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Test undocumented endpoint."}
        ]
    }
)
print("Status code:", response.status_code)
print("Raw response:", response.text)
