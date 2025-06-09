from openai import OpenAI
# import os

# port = int(os.getenv("PORT", 8000))
API_URL = f"https://aiapi-tno8.onrender.com/v1"
client = OpenAI(api_key="sk-xxx", base_url=API_URL)

if __name__ == "__main__":
    # print(port)
    print(API_URL)
    print(client)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Test undocumented endpoint."}]
    )
    print(response)

    # Correct way to inspect the response:
    print("Model:", response.model)
    print("Response content:", response.choices[0].message.content)
