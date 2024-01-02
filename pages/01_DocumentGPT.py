import os
import time
import logging
import dotenv
import pinecone
import streamlit as st
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

# Logging
logger = logging.getLogger("fgpt.documentgpt")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.handlers[0].setFormatter(formatter)
info, debug = logger.info, logger.debug

# Evironment variable
dotenv.load_dotenv(dotenv.find_dotenv("../.env"))


# Function: Embed file
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    st.write(file_content)
    st.write(file_path)
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
    debug(len(embeddings.embed_query("Hello")))


    # Pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )

    index_name = "open-ai"
    if index_name not in pinecone.list_indexes():
        # we create a new index
        info(f"Create new vector store: {index_name}")
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
        vectorstore = Pinecone.from_documents(
            docs, cahced_embeddings, index_name=index_name
        )

    else:
        info(f"Vectorstore from existing index: {index_name}")
        vectorstore = Pinecone.from_existing_index(index_name, cahced_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


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

file = st.file_uploader(
    "Upload a .txt file",
    type=["txt"],
)

if file:
    st.write(file)
    retriever = embed_file(file)
    docs = retriever.invoke("ministry of truth")
    st.write(docs)
