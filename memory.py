"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
"""

from langchain.memory import ConversationBufferWindowMemory  # Whole


memory = ConversationBufferWindowMemory(
    return_messages=True,  # for chat model
    k=4,
)


def add_message(input, output):
    global memory
    memory.save_context(
        {"input": input},
        {"output": output},
    )


for i in range(1, 6):
    add_message(i, i)

out = memory.load_memory_variables({})

print(out)
