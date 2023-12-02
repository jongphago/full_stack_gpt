"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
3. ConversationSummaryMemory
4. ConversationSummaryBufferMemory
"""

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=50,
    return_messages=True,
)


def add_message(input, output):
    global memory
    memory.save_context(
        {"input": input},
        {"output": output},
    )


def get_history():
    out = memory.load_memory_variables({})
    return out


add_message(
    "Hi I'm Jonghyun, I live in South Korea",
    "Wow that is so cool!",
)

add_message(
    "South Korea is so pretty",
    "I wish I could go!!!",
)


history = get_history()

print(history)
