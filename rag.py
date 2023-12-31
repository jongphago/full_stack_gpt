import os
import logging
import dotenv
import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

llm = ChatOpenAI()

logger = logging.getLogger("rag")
logger.setLevel(logging.DEBUG)
info, debug = logger.info, logger.debug

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=180,
)

loader = TextLoader("./files/chapter_one.txt")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cache_dir = LocalFileStore("./.cache/")
cahced_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
debug(len(embeddings.embed_query("Hello")))

# Envrionment variable
if not (success := dotenv.load_dotenv(dotenv.find_dotenv())):
    raise

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


chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_rerank",  # "refine", "map_reduce", "map_rerank"
    retriever=vectorstore.as_retriever(),  # interface
)

result = chain.run("Where does winston live")
print(result)
result = chain.run(f"Describe {result}")
print(result)
