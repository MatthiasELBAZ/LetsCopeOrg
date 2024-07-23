import nest_asyncio
nest_asyncio.apply()

import streamlit as st

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import UnstructuredReader
from pathlib import Path

import pandas as pd


st.set_page_config(page_title="Chat with the LetsCope docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Chat with the LetsCope docs, powered by LlamaIndex 💬🦙")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi Welcome to LetsCope! How can I help you today?",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # make settings
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        api_key="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"
    )

    # load data
    data_content_filepath = "./data_for_chat/data_content/df_content_description_programid.csv"
    data_content = pd.read_csv(data_content_filepath)
    data_content_documents = [
        Document(
            text=row['description'],
            metadata={
                'ID': row['id'],
                'Type': row['tipo'],
                'ProgramID': row['program_id'],
                'Author': row['autor'],
                'Title': row['title'],
                'Section': row['seccion'],
                'Category': row['categoria'],
                'URL': row['video_url'],
            }
        )
        for _, row in data_content.iterrows()
    ]

    reader = SimpleDirectoryReader(input_dir="./data_for_chat/data_article")
    data_program_documents = reader.load_data()
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=20)
    data_program_nodes = splitter.get_nodes_from_documents(data_program_documents)

    data_website_dir = Path("./data_for_chat/data_website/")
    html_page_names = []
    for file in data_website_dir.iterdir():
        if file.suffix == ".html":
            html_page_names.append(file.stem)
    loader = UnstructuredReader()
    html_page_doc_set = {}
    html_page_all_docs = []
    for html_page in html_page_names:
        html_page_docs = loader.load_data(file=Path(f"./data_for_chat/data_website/{html_page}.html"), split_documents=False)

        # insert year metadata into each year
        for d in html_page_docs:
            d.metadata = {"html_page": html_page}
        html_page_doc_set[html_page] = html_page_docs
        html_page_all_docs.extend(html_page_docs)

    # create index
    data_content_index = VectorStoreIndex.from_documents(data_content_documents)
    data_content_query_engine = data_content_index.as_query_engine()
    data_content_query_retiever = data_content_index.as_retriever(similarity_top_k=5)
    
    data_program_index = VectorStoreIndex(data_program_nodes)
    data_program_query_engine = data_program_index.as_query_engine()
    data_program_query_retriever = data_program_index.as_retriever(similarity_top_k=5)

    data_program_index = VectorStoreIndex(data_program_nodes)
    data_program_query_engine = data_program_index.as_query_engine()
    data_program_query_retriever = data_program_index.as_retriever(similarity_top_k=5)

    # create tools
    def find_top_k_docs(query: str):
        results = query_retiever.retrieve(query)
        elts = []
        for doc in results:
            doc.metadata["text"] = doc.text
            elts.append(doc.metadata)
        return elts

    find_top_k_docs_tool = FunctionTool.from_defaults(
        find_top_k_docs,
        name="find_top_k_docs",
        description="This tool is used when you have to look for content for recommendation to the user according to his mental health disconfort.",
    )

    data_content_query_engine_tool = QueryEngineTool(
        query_engine=data_content_query_engine,
        metadata=ToolMetadata(
            name="data_content_query_engine",
            description="A RAG engine with information about video for mental health made by specialists."
        )     
    )

    data_program_query_engine_tool = QueryEngineTool(
        query_engine=data_program_query_engine,
        metadata=ToolMetadata(
            name="data_program_query_engine",
            description="A RAG engine with information about programs for mental health made by specialists."
        )     
    )

    data_website_query_engine_tool = QueryEngineTool(
        query_engine=data_website_query_engine,
        metadata=ToolMetadata(
            name="data_website_query_engine",
            description="A RAG engine with information about LetsCope.org website that offer services to user for helping them to cope with mental health issues."
        )     
    )

    tools = [find_top_k_docs_tool, data_content_query_engine_tool, data_program_query_engine_tool, data_website_query_engine_tool]

    # creat openai agent
    openai_agent = OpenAIAgent.from_tools(
        tools, 
        llm=Settings.llm, 
        verbose=True,
        system_prompt="You are helpful assistant that of the website LetsCope.org. You chat with user for helping to use the website, getting information and recommendation. You can provide advices to the user according the context. You are able to recommend video based on the conversation with the user and provide the url also"
    )

    return openai_agent


openai_agent = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = openai_agent

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
