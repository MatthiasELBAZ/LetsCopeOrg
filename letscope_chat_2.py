# %%
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import QueryBundle, PromptTemplate, Document, SimpleDirectoryReader, get_response_synthesizer

from llama_index.core import VectorStoreIndex
from llama_index.core import DocumentSummaryIndex
from llama_index.core import KeywordTableIndex, SimpleKeywordTableIndex, RAKEKeywordTableIndex


from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever, SummaryIndexRetriever

from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage

from llama_index.llms.openai import OpenAI

from llama_index.agent.openai import OpenAIAgent

from llama_index.readers.file import UnstructuredReader

from typing import List

from pathlib import Path

from IPython.display import Markdown, display

import pandas as pd

# %%
# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# %%
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

# %% [markdown]
# # LLM

# %%
llm = OpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo",
    max_tokens=256
)

# %% [markdown]
# # Custom Query Engines

# %% [markdown]
# ## Content

# %%
# ingesting data
data_content_filepath = "/home/elbaz/Bureau/DataDeepDive/Lets_Cope_Freelance/data/data_content/data_program_video_content.csv"
data_content = pd.read_csv(data_content_filepath)
data_content.dropna(subset=['video_description'], inplace=True)
data_content.drop_duplicates(subset=['video_id'], inplace=True)
data_content.reset_index(drop=True, inplace=True)
data_content_documents = [
    Document(
        text=row['video_description'],
        metadata={
            'video_id': row['video_id'],
            'program_id': row['program_id'],
            'program_title': row['program_title'],
            'program_description': row['program_description'],
            'video_needs': row['video_needs'],
            'author': row['program_author'],
            'video_title': row['video_title'],
            'video_url': row['video_url'],
        }
    )
    for _, row in data_content.iterrows()
]

# create index
data_content_index = VectorStoreIndex.from_documents(data_content_documents)

# %% [markdown]
# ## Articles

# %%
class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        scores_vector_ids = {n.node.node_id: n.score for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)

            # create retrieve nodes list with adding the scores
            retrieve_nodes = [NodeWithScore(node=combined_dict[rid].node, score=scores_vector_ids[rid]) for rid in retrieve_ids]            
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

            # create retrieve nodes list with adding the scores if vector_id = keyword_id
            retrieve_nodes = []
            for rid in retrieve_ids:
                if rid in vector_ids and rid in keyword_ids:
                    retrieve_nodes.append(NodeWithScore(node=combined_dict[rid].node, score=scores_vector_ids[rid]))
                elif rid in vector_ids:
                    retrieve_nodes.append(NodeWithScore(node=combined_dict[rid].node, score=scores_vector_ids[rid]))
                elif rid in keyword_ids:
                    retrieve_nodes.append(combined_dict[rid])
            
        return retrieve_nodes

# %%
# create query engine promptes
text_qa_template = (
    "You are taksed with helping the user to cope with their mental health.\n"
    "You can use context information is below from articles written by specialists.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge answer the query.\n"
    "If the context is not useful, inform the user that you don't have enough information to answer the query and should contact specialist or medical center.\n"
    "Query: {query_str}\n"
    "Answer: "
)
text_refine_template = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

# params
input_dir = "./data/data_articles/"
chunk_size = 128
chunk_overlap = 8
prompt_qa_template = PromptTemplate(text_qa_template)
prompt_refine_template = PromptTemplate(text_refine_template)
similarity_top_k = 10
similarity_cutoff = 0.7
keyword_retriever_mode = 'default'
vector_keyword_retriever_mode = "AND"
query_engine_response_mode = "default"

# ingest data
reader = SimpleDirectoryReader(input_dir=input_dir)
data_article_documents = reader.load_data()

# split documents into nodes
splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
data_article_nodes = splitter.get_nodes_from_documents(data_article_documents)

# create vector index
data_article_vector_index = VectorStoreIndex(data_article_nodes)

# configure vector retriever
data_article_vector_retriever = VectorIndexRetriever(
    index=data_article_vector_index,
    similarity_top_k=similarity_top_k,
)

# create keyword table index
if keyword_retriever_mode == 'default':
    data_article_keyword_index = KeywordTableIndex(data_article_nodes)

if keyword_retriever_mode == 'simple':
    data_article_keyword_index = SimpleKeywordTableIndex(data_article_nodes)

if keyword_retriever_mode == 'rake':
    data_article_keyword_index = RAKEKeywordTableIndex(data_article_nodes)

# configure keyword retriever
data_article_keyword_retriever = KeywordTableSimpleRetriever(
    index=data_article_keyword_index,
    similarity_top_k=similarity_top_k,
    retriever_mode=keyword_retriever_mode
)

# create custom retriever
data_article_retriever = CustomRetriever(
    vector_retriever=data_article_vector_retriever,
    keyword_retriever=data_article_keyword_retriever,
    mode=vector_keyword_retriever_mode
)

# configure response synthesizer
data_article_response_synthesizer = get_response_synthesizer(
    text_qa_template=prompt_qa_template, 
    refine_template=prompt_refine_template
)

# create query engine
data_article_query_engine = RetrieverQueryEngine.from_args(
    retriever=data_article_retriever,
    response_synthesizer=data_article_response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
    response_mode=query_engine_response_mode
)


# %% [markdown]
# ## Website

# %%
# create query engine promptes
text_qa_template = (
    "You are taksed with answering user queries about LetsCope.org though its website content.\n"
    "You can use the context information is below from website html pages.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge answer the query to advise user.\n"
    "If the context is not useful, inform the user that you don't have enough information to answer the query and should start a trial account at LetsCope.org and discover the company.\n"
    "Query: {query_str}\n"
    "Answer: "
)
text_refine_template = (
    "The original query is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
)

# params
prompt_qa_template = PromptTemplate(text_qa_template)
prompt_refine_template = PromptTemplate(text_refine_template)
input_dir = "./data/data_website/"
chunk_size = 128
chunk_overlap = 8
similarity_top_k = 10
query_engine_response_mode = "default"




# %%
# ingest data
data_website_dir = Path(input_dir)
html_page_names = []
for file in data_website_dir.iterdir():
    if file.suffix == ".html":
        html_page_names.append(file.stem)
        
loader = UnstructuredReader()
html_page_doc_set = {}
data_website_documents = []
for html_page in html_page_names:
    html_page_docs = loader.load_data(file=Path(input_dir+f"{html_page}.html"), split_documents=False)
    for d in html_page_docs:
        d.metadata = {"html_page": html_page}
    html_page_doc_set[html_page] = html_page_docs
    data_website_documents.extend(html_page_docs)


# split documents into nodes
splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)



# %%
# # configure summary response synthesizer
# data_website_summary_synthesizer = get_response_synthesizer(
#     response_mode="tree_summarize", 
#     use_async=True
# )

# # create summary index
# data_website_summary_index = DocumentSummaryIndex.from_documents(
#     data_website_documents,
#     llm=llm,
#     transformations=[splitter],
#     response_synthesizer=data_website_summary_synthesizer,
#     show_progress=False,
# )

# # create summary retriever
# data_website_summary_retriever = data_website_summary_index.as_retriever(similarity_top_k=1)

# # configure response synthesizer
# data_website_response_synthesizer = get_response_synthesizer(
#     text_qa_template=prompt_qa_template, 
#     refine_template=prompt_refine_template
# )

# # create query engine
# data_website_summary_query_engine = RetrieverQueryEngine.from_args(
#     retriever=data_website_summary_retriever,
#     response_synthesizer=data_website_response_synthesizer,
#     response_mode=query_engine_response_mode
# )

# %%
data_website_nodes = splitter.get_nodes_from_documents(data_website_documents)

# create vector index
data_website_vector_index = VectorStoreIndex(data_website_nodes)

# configure vector retriever
data_website_vector_retriever = VectorIndexRetriever(
    index=data_website_vector_index,
    similarity_top_k=10,
)

# configure response synthesizer
data_website_response_synthesizer = get_response_synthesizer(
    text_qa_template=prompt_qa_template, 
    refine_template=prompt_refine_template
)

# create query engine
data_website_vector_query_engine = RetrieverQueryEngine.from_args(
    retriever=data_website_vector_retriever,
    response_synthesizer=data_website_response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    response_mode=query_engine_response_mode
)

# %% [markdown]
# # Tools

# %%
def find_top_k_docs(query: str, k: int):
    data_content_query_retiever = data_content_index.as_retriever(similarity_top_k=k)
    results = data_content_query_retiever.retrieve(query)
    elts = []
    for doc in results:
        doc.metadata["text"] = doc.text
        elts.append(doc.metadata)
    return elts

find_top_k_docs_tool = FunctionTool.from_defaults(
    find_top_k_docs,
    name="find_top_k_docs",
    description="This tool is used when you think it s relevant to recommand a video content to the user or when the user is asking for video recommendation.",
)

# %%
data_article_query_engine_tool = QueryEngineTool(
    query_engine=data_article_query_engine,
    metadata=ToolMetadata(
        name="data_article_query_engine",
        description="A RAG engine with information about articles for coping with mental health disconfort made by specialists."
    )     
)

# %%
data_website_query_engine_tool = QueryEngineTool(
    query_engine=data_website_query_engine,
    metadata=ToolMetadata(
        name="data_website_query_engine",
        description="A RAG engine with information about LetsCope.org website that offer services to user for helping them to cope with mental health issues."
    )     
)

# %% [markdown]
# # Chat Agent

# %%
system_prompt = """
    You are helpful assistant of the company LetsCope.org. 

    You are tasked to chat with user and answering his queries about the company LetsCope.org and its services, his mental health issues and how to cope with them.
        
    You can provide advices to the user according mental health coping articles posted by specialists. 
    
    You are able to recommend video based on the conversation with the user and provide the url also.
"""

# %%
tools = [find_top_k_docs_tool, data_article_query_engine_tool, data_website_query_engine_tool]

# %%
openai_agent = OpenAIAgent.from_tools(
    tools=tools, 
    llm=llm, 
    verbose=True,
    system_prompt=system_prompt
)

# %%
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    print(user_input)
    response = openai_agent.chat(user_input)
    print(f"Assistant: {response.response}")
    print()

# %%



