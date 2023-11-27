from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector


class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        from random import choice

        return [choice(self.examples)]


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
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=80,
)
example_selector = RandomExampleSelector(
    examples=examples,
)
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    suffix="Human: What do you know about {country}?",
    input_variables=["country"],
)

format = prompt.format(country="Korean")
print(format)

chain = prompt | chat

chain.invoke({"country": "Korea"})
