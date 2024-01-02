import os
import time
import logging
import dotenv
import pinecone
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

# Evironment variable
dotenv.load_dotenv(dotenv.find_dotenv("../.env"))


# Function: Embed file
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=180,
    )

    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cache_dir = LocalFileStore(f".cache/embeddings/{file.name}")
    cahced_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "open-ai"
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        vectorstore = Pinecone.from_documents(
            docs, cahced_embeddings, index_name=index_name
        )

    else:
        vectorstore = Pinecone.from_existing_index(index_name, cahced_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if "message" not in st.session_state:
        st.session_state["message"] = []
    if save:
        st.session_state["message"].append(
            {
                "message": message,
                "role": role,
            }
        )


def paint_history():
    for message in st.session_state["message"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(temperature=0.1)

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!
"""
)
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
    )

if file:
    retriever = embed_file(file)
    message = st.chat_input("")
    send_message("I'm ready! Ask away", "AI", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        time.sleep(1)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "AI")

else:
    st.session_state["message"] = []
