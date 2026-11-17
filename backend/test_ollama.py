import ollama

response = ollama.chat(
    model="llama3.1",
    messages=[
        {
            "role": "user",
            "content": "Summarize this customer feedback in one sentence: The product is excellent but delivery was late."
        }
    ]
)

print("\nAI Response:")
print(response["message"]["content"])