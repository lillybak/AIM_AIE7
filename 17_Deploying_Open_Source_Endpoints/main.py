def main():
    print("Hello from 17-deploying-open-source-endpoints!")
    from langchain_together import ChatTogether

    llm = ChatTogether(
        model="meta-llama/Llama-3-70b-chat-hf",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )

    messages = [
        ("system", "You are a helpful assistant that translates English to French. Translate the user sentence."),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)

if __name__ == "__main__":
    main()
