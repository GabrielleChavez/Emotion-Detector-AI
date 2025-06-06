from google import genai

client = genai.Client(api_key="AIzaSyA1bSl6VT0YXC9tujsY-E6gkbJW3sippV0")

ask = input("Enter your question: ")

while ask != "exit":

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=ask,
    )

    print(response.text)
    ask = input("Enter your question: ")