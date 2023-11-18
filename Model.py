from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate


chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

examples = [
    {
        "question": "What do you know about France?",
        "answer": """
    Here is what I know:
    Capital: Paris
    Language: French
    Food: Wine and Cheese
    Currency: Euro
    """,
    },
    {
        "question": "What do you know about Italy?",
        "answer": """
    I know this:
    Capital: Rome
    Language: Italian
    Food: Pizza and Pasta
    Currency: Euro
    """,
    },
    {
        "question": "What do you know about Greece?",
        "answer": """
    I know this:
    Capital: Athens
    Language: Greek
    Food: Souvlaki and Feta Cheese
    Currency: Euro
    """,
    },
]

example_prompt = PromptTemplate.from_template("Human: {question}\nAI: {answer}")
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

chain = prompt | chat

chain.invoke({"country": "Korea"})
