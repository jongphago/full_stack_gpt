from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.globals import set_llm_cache, set_debug
from langchain.cache import InMemoryCache, SQLiteCache
from time import time

# set_llm_cache(InMemoryCache())
set_llm_cache(SQLiteCache("cache.db"))
set_debug(True)

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

begin = time()
chat.predict("How do you make italian pasta")
end1 = time() - begin

begin = time()
chat.predict("How do you make italian pasta")
end2 = time() - begin

print(end1)  # 12.528858184814453
print(end2)  # 0.010240554809570312
