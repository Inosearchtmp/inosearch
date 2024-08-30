# __import__('pysqlite3')
import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
import os

from llama_index.llms.huggingface import HuggingFaceLLM

def messages_to_prompt(messages):
    prompt = ""
    system_found = False
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
            system_found = True
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"
        else:
            prompt += f"<|user|>\n{message.content}<|end|>\n"

    # trailing prompt
    prompt += "<|assistant|>\n"

    if not system_found:
        prompt = (
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
        )

    return prompt


llm = HuggingFaceLLM(
    model_name="microsoft/Phi-3-mini-128k-instruct",
    model_kwargs={
        "trust_remote_code": True,
    },
    generate_kwargs={"do_sample": True, "temperature": 0.1},
    tokenizer_name="microsoft/Phi-3-mini-128k-instruct",
    query_wrapper_prompt=(
        "<|system|>\n"
        "You are a helpful AI assistant.<|end|>\n"
        "<|user|>\n"
        "{query_str}<|end|>\n"
        "<|assistant|>\n"
    ),
    messages_to_prompt=messages_to_prompt,
    is_chat_model=True,
)