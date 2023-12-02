"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
3. ConversationSummaryMemory
4. ConversationSummaryBufferMemory
5. ConversationKGMemory
"""

from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0.1)

memory = ConversationKGMemory(
    llm=llm,
    return_messages=True,
)


def add_message(input, output):
    global memory
    memory.save_context(
        {"input": input},
        {"output": output},
    )


def get_history(input):
    out = memory.load_memory_variables({"input": input})
    return out


add_message(
    "Hi I'm Jonghyun, I live in South Korea",
    "Wow that is so cool!",
)

add_message(
    "South Korea is so pretty",
    "I wish I could go!!!",
)


history = get_history("Who is Jonghyun")

print(history)
