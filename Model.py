from langchain.llms.openai import OpenAI

chat = OpenAI(temperature=0.1, max_tokens=450, model="gpt-3.5-turbo-16k")

chat.save("model.json")
