"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
3. ConversationSummaryMemory
4. ConversationSummaryBufferMemory
5. ConversationKGMemory
"""

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory


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


chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True,
)

chat = chain.predict(question="My name is Jonghyun")
print(chat)

chat = chain.predict(question="I live in Seoul")
print(chat)

out = chain.predict(question="What is my name?")
print(out)
