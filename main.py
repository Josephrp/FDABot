import streamlit as st

import json
from typing import Any, Dict, List, Optional, Generator
import os

import transformers
from transformers import AutoTokenizer
from src.engine.prompts import systemprompt
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.readers.json import JSONReader
# from src.tools.webpagescrapper import WebScrapper
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from llama_index.core import VectorStoreIndex
# from src.tools.webpagescrapper import WebScrapper
# from src.utils.json_handler import load_json_file
# from llama_index.readers.JSONReader import JSONReader
# from llama_index.readers.json.base import JSONReader
from llama_index.core.schema import Document
from typing import Any, Dict, Generator, List, Optional
# import src.config.config
from dotenv import load_dotenv

from src.config.config import Settings
from llama_index.core import StorageContext, load_index_from_storage

load_dotenv()
# print(st.secrets["REPLICATE_API_TOKEN"])
# os.environ["REPLICATE_API_TOKEN"] == st.secrets["REPLICATE_API_TOKEN"]
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="storage")

Settings.llm = Replicate(
    model="snowflake/snowflake-arctic-instruct",
    )

# load index
index = load_index_from_storage(storage_context)

persistant_query_agent = index.as_query_engine(
    llm=Settings.llm,
    )

# Set assistant icon 
icons = {"assistant": "🏛️", "user": "⛷️"}

# App title
# st.set_page_config(page_title="FDABot - Your FDA Website Knowledge Partner")

# Sidebar - Replicate Credentials (Removed external API usage for this version)
# os.environ['REPLICATE_API_TOKEN'] = 'Your Replicate Token Here'  # If needed
# We will skip the API token part for now since the query engine will be used.

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": systemprompt}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=icons[message["role"]]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": systemprompt}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


# def generate_response():
#     prompt = []
#     for dict_message in st.session_state.messages:
#         if dict_message["role"] == "user":
#             prompt.append(dict_message["content"])
    
#     prompt_str = " ".join(prompt)
    
#     if get_num_tokens(prompt_str) >= 3072:
#         st.error("Conversation length too long. Please keep it under 3072 tokens.")
#         st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
#         st.stop()

#     response = persistant_query_agent.query(prompt_str)
#     print(response)
#     return str(response)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="⛷️"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar="🏛️"):
        # response = generate_response()
        print(st.session_state.messages[-1]['content'])
        last_output = st.session_state.messages[-1]['content']
        response = persistant_query_agent.query(last_output)
        # full_response = st.write(response)
        print(response)
        st.write(response.response)
        # st.write(response)
        # print(message)
    message = {"role": "assistant", "content": response}
    # st.session_state.messages.append(message)
    # st.session_state.messages.append(message['reponse'])
    st.session_state.messages.append(message)
