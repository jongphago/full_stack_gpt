"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
3. ConversationSummaryMemory
4. ConversationSummaryBufferMemory
5. ConversationKGMemory
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough


llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    return_messages=True,
    memory_key="chat_history",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]


chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm

base_message = chain.invoke(
    {
        "question": "My name is Jonghyun",
    }
)

print(base_message)
