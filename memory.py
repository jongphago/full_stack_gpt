"""
1. WholeConversationalMemory
"""

from langchain.memory import ConversationBufferMemory  # Whole


memory = ConversationBufferMemory(
    return_messages=True,  # for chat model
)
memory.save_context(
    {"input": "Hi!"},
    {"output": "How are you?"},
)

out = memory.load_memory_variables({})

print(out)
