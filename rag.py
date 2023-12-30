from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    seperator="\n",
    chunk_size=600,
    chunk_overlab=180,
)

loader = TextLoader("./files/chapter_one.txt")
