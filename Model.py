from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from numpy import character


chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

intro = PromptTemplate.from_template(
    """Yor are a role playing assistant.
    And you are impersonating a {character}
    """
)

example = PromptTemplate.from_template(
    """This is an example of how you talk:
    
    Human: {example_question}
    You: {example_answer}
    """
)

start = PromptTemplate.from_template(
    """Start now!
    
    Human: {question}
    You:
    """
)

final = PromptTemplate.from_template(
    """{intro}

    {example}
    
    {start}
    """
)

prompts = [
    ("intro", intro),
    ("example", example),
    ("start", start),
]
full_prompt = PipelinePromptTemplate(final_prompt=final, pipeline_prompts=prompts)

format = full_prompt.format(
    character="Pirate",
    example_question="What is your location?",
    example_answer="Arrrg! That is a secret!! arg arg!!",
    question="What is your favoriate food?",
)
print(format)

chain = full_prompt | chat
chain.invoke(
    {
        "character": "Pirate",
        "example_question": "What is your location?",
        "example_answer": "Arrrg! That is a secret!! arg arg!!",
        "question": "What is your favoriate food?",
    }
)
