__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)

from llama_index.llms.huggingface import HuggingFaceLLM

import os
import yaml
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

# Choose the embedding model based on the configuration
if config["embedding_model"]["type"] == "openai":
    embed_model = OpenAIEmbedding(model=config["embedding_model"]["model_name"])
elif config["embedding_model"]["type"] == "huggingface":
    embed_model = HuggingFaceEmbedding(model_name=config["embedding_model"]["model_name"], 
                                       trust_remote_code=config["embedding_model"]["trust_remote_code"])
else:
    raise ValueError("Unsupported embedding model type specified in the config file.")

# Apply settings
Settings.embed_model = embed_model
Settings.chunk_size = config["chunk_size"]




def create_index(documents_folder_path, vector_db_path):
    documents = SimpleDirectoryReader(documents_folder_path).load_data()

    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path=vector_db_path)

    # create collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

def load_index(vector_db_path):
    
    # initialize client
    db = chromadb.PersistentClient(path=vector_db_path)

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    return index


def answer_query(query,vector_db_path):
    anthropic_api_key = config['anthropic_api_key']
    os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
    llm = Anthropic(temperature=0.0, model='claude-3-haiku-20240307')

    Settings.llm = llm

    index = load_index(vector_db_path)
    query_engine = index.as_query_engine(similarity_top_k=3)

    prompt = """Please provide the information requested and use all the documents provided, and for each piece of information,
      include the name of the document source in brackets immediately after the information.
        The name of the document source should be concise and placed within brackets like this: [DocumentName]."""


    response = query_engine.query(query+prompt)
    return response


