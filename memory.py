"""
1. WholeConversationalMemory
2. ConversationBufferWindowMemory
3. ConversationSummaryMemory
4. ConversationSummaryBufferMemory
5. ConversationKGMemory
"""

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory


llm = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history"
)

template = """
    You are a helpful AI talking to a human.
    
    {chat_history}
    Human: {question}
    You: 
"""

chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template(template),
    verbose=True,
)


def add_message(input, output):
    global memory
    memory.save_context(
        {"input": input},
        {"output": output},
    )


chat = chain.predict(
    question="Hi I'm Jonghyun, I live in South Korea",
)
print(chat)

out = chain.predict(
    question="What is my name?",
)
print(out)

history = memory.load_memory_variables({})
print(history)
