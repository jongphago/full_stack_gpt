from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

chat = ChatOpenAI(
    temperature=0.1,
)

with get_openai_callback() as usage:
    chat.predict("What is the recipe for soju")
    print(usage)
