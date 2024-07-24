import streamlit as st
from openai import OpenAI
import json
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import Document
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, KGTableRetriever, VectorIndexRetriever
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle, NodeWithScore
from typing import List

# Show title and description.
st.title("Amazon Reviews Graph Question Answering")
st.write("Enter the question about reviews on Amazon.com in the health care category. e.g. What do users think of electric shavers?")
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.text_input("Enter OpenAI Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client.
    os.environ["OPENAI_API_KEY"] = openai_api_key
    OPENAI_API_KEY = openai_api_key

# Ask the user for a question via `st.text_area`.
num_asins = st.text_area(
    "How many ASINs to use?",
)

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(".", 'db.sqlite3'),
    }
}

file_reviews = "Health_and_Personal_Care.jsonl"
reviews = dict()
already_added = set()
with(open(file_reviews, "r") as f):
    for val in f:
        js = json.loads(val)
        text = js["text"]
        asin = js["parent_asin"]
        title = js["title"]
        rating = js["rating"]
        userid = js["user_id"]
        st1 = f"""ASIN: "{asin}", Title: "{title}", Review Text: "{text}", Rating: {rating} out of 5, User: {userid}.\n\n"""
        if st1 not in already_added:
            already_added.add(st1)
            if asin in reviews:
                reviews[asin].append(st1)
            else:
                l = [st1]
                reviews[asin] = l

docs = dict()
count = 0
num_reviews = 0
for key in reviews:
    r = reviews[key]
    one_document_text = ""
    for val in r:
        num_reviews+=1
        one_document_text = one_document_text + val + "\n\n"
    docs[key] = one_document_text
    count+=1
    if count > int(num_asins):
        break

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
text_splitter = SemanticChunker(OpenAIEmbeddings())

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
ldocs = [Document(page_content=val, metadata={"source": "local"}) for val in docs.values()]
db_chroma = FAISS.from_documents(ldocs, OpenAIEmbeddings())

"""
for key in docs.keys():
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name=key)
    doc_texts = docs[key]
    collection.add(
        documents=[doc_texts],
        ids=[str(hash(t)) for t in [doc_texts]]
    )
    db_chroma[key] = collection
"""

title_to_asin = dict()
titles = []
file_meta = "meta_Health_and_Personal_Care.jsonl"
with(open(file_meta, "r") as f):
    for val in f:
        js = json.loads(val)
        if js["parent_asin"] in docs.keys():
            titles.append(js["title"])
            title_to_asin[js["title"]] = js["parent_asin"]

import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="review_titles_new_250")
collection.add(
    documents=titles,
    ids=[str(hash(t)) for t in titles]
)

from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def completion(prompt: str, model_name: str) -> str:
    chat = ChatOpenAI(temperature=0, model_name=model_name, openai_api_key=OPENAI_API_KEY)
    messages = [
        HumanMessage(
            content=prompt
        )
    ]
    return chat(messages).content

def create_entire_prompt_three_step(system, text):
    prompt = f"""
    System: {system}

    User:
    --- Context Begin ---
    {text}
    --- Context End ---

    --- Instructions Begin ---
    Your task is to generate important connections between entities (facts) from the context and title above.\n
    Please use a six step process:\n

    1. Use open information extraction to extract all entities and relations.\n
    2. Create a schema for the knowledge graph, but don't output the schema. While creating a schema, keep in mind the fact that these relations will be used to construct a knowledge graph.\n
    3. Create canonical relations - Create simplified relations and combine similar relations to the same name of the relation.\n
    4. Disambiguate named entities - Two entitites that have a strong relation will have similar behaviors. e.g. Elon Musk and Tesla, or Bill Gates and Microsoft. Both entities should be considered a single named entity.\n
    5. Combine behaviors. For example, if the text says "Elon Musk's Apple increased prices of iphones in China", both Elon Musk and Apple have the behavior of increasing prices of iphones and the entity should be "Elon Musk's Apple increased iphone prices".\n
    6. Go deep into the news article and reflect on all connections between entitites. Any spurious connections should be removed.
    7. Pay special attention to the tense of the verbs. e.g. "Apple had invested 10 billion in security stocks but sold off most of the investment". This means that a connection (Apple, invested in, security stocks) is not valid in the present. The connection should be (Apple, had invested in, security stocks).\n
    8. When you merge connections, please make sure you remove the duplicate connections.\n\n
    9. There should be no duplicate entities and connections. If the edges are duplicates, please remove the duplicates. Also, ensure that there are a maximum of 50 edges and connections.\n
    10. Pay special attention to years and dates. e.g. if the sentence is "Apple invested 10 million in generative AI in 2024", then include the 2024 in the edge (connection) or the entities.\n\n

    Pay special attention to verbs like "increased", "decreased", "profit", "loss" etc. These verbs are important to get a summary of how the company is doing in the market.\n

    Please output entities in both directions e.g. if the connection is (Microsoft, invested, 10B in the AI field), please output the connection in the other direction as well e.g. (10B in the AI field, was invested by, Microsoft)\n\n

    The task is to generate important facts that the context mentions in JSON format of the form of \n
    [{{"ENTITY_1":"entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2" }}, {{"ENTITY_2" : "entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2" }}, {{"ENTITY_1" : "entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2"}} and so on for all entity connections....\n
    The entities (entity_1, entity_2) and the fact (connection_type) should be summarized to 1, 2 or 3 words using abstractive summarization.\n
    For example, (Walmart, revenue, 10B) or (Target, established, 1954) or (Diwali, celebrated by, Hindus).\n
    Please extract a minimum of 10 facts and connections and a maximum of the top 50 facts and connections from the context.\n
    Take a deep breath, and think step by step, read the full context and title, but do not return the thought process in the summary.\n
    The entities (entity_1, entity_2) and the fact (connection_type) should be summarized to 1, 2 or 3 words using abstractive summarization.\n
    Don't include generalist connection_type values, please be as specific as possible. For example, if the extracted connection_type is 'role', use the actual value of the role, like president, vice president, engineer etc.\n
    Also, don't omit the actual numbers from the entities. For example, if the extracted connection_type is 'market share', include the actual market share, like 25%, increasing, decreasing etc.\n
    Please make sure that the entities are one to four words in length. If necessary, please use abstractive summarization to reduce the length of the entities.\n
    Also, if the entity is a common noun, like viewership, please make sure to include the proper noun in ENTITY_1 or ENTITY_2. For example, if the connection is viewership, you should include the propernoun such as Netflix. So, the entity would be "Netflix Viewership" instead of just "Viewership".\n
    The ENTITY_1, CONNECTION and ENTITY_2 should be self-contained and it should be able to understand the context of the connection without reading the context.\n
    Please make sure that ENTITY_1 and ENTITY_2 are independent single entities. For example do not use "and" to include two entities as a single entity. e.g. (Microsoft and Google, invested in, Artificial Intelligence) should be (Microsoft, invested in, Artificial Intelligence) and (Google, invested in, Artificial Intelligence)\n
    Please ensure that the output is formatted as valid JSON. Please use double quotes in the JSON output.\n
    Please make sure that you produce output. An empty output is not acceptable.\n
    --- Instructions End ---

    Assistant:
    """
    return prompt

def get_system_message():
    return """You are an expert at finding named entities and the connections (facts) between them to generate a knowledge graph from text (in JSON format).\n
    The task is to generate important facts that the context mentions in JSON format of the form of \n
    [{{"ENTITY_1":"entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2" }}, {{"ENTITY_2" : "entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2" }}, {{"ENTITY_1" : "entity_1", "CONNECTION": "connection_type", "ENTITY_2" : "entity_2"}} and so on for all entity connections....\n
    Please output entities in both directions e.g. if the connection is (Microsoft, invested, 10B in the AI field), please output the connection in the other direction as well e.g. (10B in the AI field, was invested by, Microsoft)\n
    Please make sure that the entities are one to four words in length. If necessary, please use abstractive summarization to reduce the length of the entities.\n
    Please extract a minimum of 10 facts and connections and a maximum of the top 50 facts and connections from the context.\n
    Please make sure that ENTITY_1 and ENTITY_2 are independed single entities. For example do not use "and" to include two entities as a single entity. e.g. (Microsoft and Google, invested in, Artificial Intelligence) should be (Microsoft, invested in, Artificial Intelligence) and (Google, invested in, Artificial Intelligence)\n
    Don't include generalist connection_type values, please be as specific as possible. For example, if the extracted connection_type is 'role', use the actual value of the role, like president, vice president, engineer etc.\n
    Also, don't omit the actual numbers from the entities. For example, if the extracted connection_type is 'market share', include the actual market share, like 25%, increasing, decreasing etc.\n\n
    Pay special attention to years and dates. e.g. if the sentence is "Apple invested 10 million in generative AI in 2024", then include the 2024 in the edge (connection) or the entities.\n
    Please pay special attention to the tense. For example, there is a difference between the connection_type "had invested" and "invested".\n
    Pay special attention to verbs like "increased", "decreased", "profit", "loss" etc. These verbs are important to get a summary of how the company is doing in the market.\n
    Please make sure that there are no duplicate edges and connections. If there are duplicates, please remove the duplicates.\n
    Please make sure that the output is formatted as valid JSON. Please use double quotes in the JSON output.\n"""

def get_query_prompt(query, context, rag_text, vector_text):
    return f"""

    --- Vector Index Retrieved Document Chunks Begin ---
    {vector_text}
    --- Vector Index Retrieved Document Chunks End ---

    --- Knowledge Graph Triples Context Begin ---\n
    {context}\n
    --- Knowledge Graph Triples Context End ---\n\n

    --- Document Context Begin ---
    {rag_text}\n
    --- Document Context End ---

    --- Query Begin ---\n
    {query}\n
    --- Query End ---\n\n

    --- Instructions Begin ---
    Please answer the query above using the context provided above.\n
    Please be very verbose in the answers.\n
    Pay special attention to the surrounding text to see if the numbers are stock prices, revenues.\n
    Please distinguish between the revenue and stock prices. For instance $ 10B might be a revenue while $ 500 might be a stock price.\n
    Don't get confused between revenue, stock prices, profit and income.\n
    Pay special attention to the descriptions of the numbers in the surrounding text.\n
    Generate a verbose answer to the query above.\n
    Take a deep breath, think step by step but don't answer the thought process.\n
    Go deep into the text and the triples, and reflect on the answers for accuracy.\n
    Reply only with the answer to the query. Don't output anything else.\n
    Feel free to say that the report does not include information to answer the query.\n
    Please do not output the source of the information, like the knowledge graph triples or the document.\n
    Please keep in mind that the reader of the output is an investment analyst.\n
    --- Instructions End ---
    """
def get_query_system_prompt():
    return """
    You are an expert question answering system. You use the knowledge graph triples as context and answer the questions in a verbose manner.\n
    Please be very verbose in the answers.\n
    Pay special attention to the surrounding text to see if the numbers are stock prices, revenues.\n
    Please distinguish between the revenue and stock prices. For instance $ 10B might be a revenue while $ 500 might be a stock price.\n
    e.g. $125 is a stock price. $10B is revenue, profit or income. $100,000M is revenue, profit or income.\n
    e.g. $350 is a stock price. $100 B is revenue, profit or income. $250,000 M is revenue, profit or income.\n
    Pay special attention to the descriptions of the numbers in the surrounding text.\n
    Generate a verbose answer to the query above.\n
    Take a deep breath, think step by step but don't answer the thought process.\n
    Go deep into the text and the triples, and reflect on the answers for accuracy.\n
    Reply only with the answer to the query. Don't output anything else.\n
    Feel free to say that the report does not include information to answer the query.\n
    Please do not output the source of the information, like the knowledge graph triples or the document.\n
    Please keep in mind that the reader of the output is an investment analyst.\n
    """

model_test_1 = 'ft:gpt-3.5-turbo-0125:anvai-ai::9SBH6Gog'
model_test_2 = 'ft:gpt-3.5-turbo-0125:anvai-ai::9UjMtBU8'
model_control = 'gpt-3.5-turbo'
model_gpt4 = 'gpt-4'
model_gpt4o = "gpt-4o"

import json
import statistics
import random
system_message = get_system_message()

def create_kg(docs):
    entire_json_key = []
    count = 0
    count+=1
    text = docs
    entire_prompt_three_step = create_entire_prompt_three_step(system_message, text)
    completed_text_fine_tuned_model = completion(entire_prompt_three_step, model_test_2)
    print(count)
    print(completed_text_fine_tuned_model)
    try:
        js = json.loads(completed_text_fine_tuned_model)
        entire_json_key.extend(js)
    except:
        print("Could not load json from this chunk. Ignoring!")

    return entire_json_key

count = 0
entire_json = dict()
for key in docs:
    file = "./graph_json/" + key + ".json"
    if not os.path.exists(file):
        doc = docs[key]
        if len(doc.split()) > 10000:
            continue
        print(count)
        count+=1
        try:
            entire_json_key = create_kg(doc)
            open(file, "w").write(json.dumps(entire_json_key))
            entire_json[key] = entire_json_key
        except:
            print("Too large prompt! Ignoring.")
    else:
        print(count)
        count+=1
        s = open(file, "r").read()
        entire_json[key] = json.loads(s)

from scipy.sparse import linalg
from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.schema import BaseNode, IndexNode, TextNode
import os
import sys

llm_openai = OpenAI(model=model_test_2, temperature=0)
Settings.llm = llm_openai
llm_gpt4o = OpenAI(model=model_gpt4o, temperature=0)
Settings.embed_model = OpenAIEmbedding()

import kuzu
from llama_index.graph_stores.kuzu import KuzuGraphStore

index = dict()
for key in docs.keys():
    try:
        db_key = kuzu.Database("./kuzu_databases/" + key)
        graph_store = KuzuGraphStore(db_key)
        storage_context_key = StorageContext.from_defaults(graph_store=graph_store)
        index_key = KnowledgeGraphIndex(
            [],
            storage_context=storage_context_key,
            include_embeddings=False,
            llm=llm_openai
        )

        index[key] = index_key
    except:
        print("Database storing failed for Kuzu! Ignoring.")

def add_to_kg(index, js):
    nodekeywords = dict()
    for val in js:
        if "ENTITY_1" not in val or "ENTITY_2" not in val or "CONNECTION" not in val:
            continue

        n1 = val["ENTITY_1"]
        n2 = val["ENTITY_2"]
        c = val["CONNECTION"]

        if n1 == "" or n2 == "" or c == "":
            continue

        tup1 = [n1, c, n2]
        print(tup1)
        keywords = []
        for val in n1.split():
            keywords.append(val)

        for val in n2.split():
            keywords.append(val)

        for val in c.split():
            keywords.append(val)

        keywords.append(n1)
        keywords.append(n2)
        keywords.append(c)

        n1k = set()
        if n1 in nodekeywords:
            n1k.update(keywords)
            n1k.update(nodekeywords[n1])
            nodekeywords[n1] = n1k
        else:
            n1k.update(keywords)
            nodekeywords[n1] = n1k

        n2k = set()
        if n2 in nodekeywords:
            n2k.update(keywords)
            n2k.update(nodekeywords[n2])
            nodekeywords[n2] = n2k
        else:
            n2k.update(keywords)
            nodekeywords[n2] = n2k

        node1 = TextNode()
        node1.set_content(n1)
        index.add_node(list(n1k), node1)

        node2 = TextNode()
        node2.set_content(n2)
        index.add_node(list(n2k), node2)

        try:
            index.upsert_triplet_and_node(tup1, node1, include_embeddings=False)
            index.upsert_triplet_and_node(tup1, node2, include_embeddings=False)
        except:
            print("Error, Ignoring the triple!")
            continue

for key in index.keys():
    index_key = index[key]
    if key not in entire_json:
        continue

    if not os.path.exists("./kuzu_databases/" + key):
        js = entire_json[key]
        add_to_kg(index_key, js)

doc_dict = dict()
for key in docs.keys():
    if key not in entire_json:
        continue

    js = entire_json[key]
    doc_list = []
    for val in js:
        if "ENTITY_1" not in val or "ENTITY_2" not in val or "CONNECTION" not in val:
            continue

        n1 = val["ENTITY_1"]
        n2 = val["ENTITY_2"]
        c = val["CONNECTION"]

        if n1 == "" or n2 == "" or c == "":
            continue

        tup1 = (n1, c, n2)
        doc_list.append(Document(text=str(tup1)))

    doc_dict[key] = doc_list

from llama_index.core import StorageContext, load_index_from_storage

vector_index = dict()
for key in docs.keys():
    if key not in entire_json:
        continue

    if os.path.exists("./vectors/" + key + "_VECTORS"):
        storage_context = StorageContext.from_defaults(persist_dir="./vectors/" + key + "_VECTORS")
        index_key = load_index_from_storage(storage_context)
    else:
        documents_key = doc_dict[key]
        index_key = VectorStoreIndex.from_documents(documents_key)
        index_key.storage_context.persist(persist_dir="./vectors/" + key + "_VECTORS")

    vector_index[key] = index_key

query_engine_vectors = dict()
query_engine_keywords = dict()

for key in docs:
    if key in index:
        query_engine_key_keyword = index[key].as_query_engine(
            response_mode="tree_summarize",
            verbose=False,
            llm=llm_openai,
            include_text=False,
            max_keywords_per_query=50,
            retriever_mode="keyword", # change this to "keyword" or "hybrid" for spreadsheet querying
            use_global_node_triplets=True,
            max_knowledge_sequence=500,
            num_chunks_per_query=5,
            )

        query_engine_keywords[key] = query_engine_key_keyword

    if key in vector_index:
        query_engine_key_vector = vector_index[key].as_query_engine(
            response_mode="tree_summarize",
            max_knowledge_sequence=500,
            similarity_top_k=50,
            )

        query_engine_vectors[key] = query_engine_key_vector

queries = []
# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
)

queries.append(question)
res_str = ""
for val in queries:
    results = collection.query(
        query_texts=[val], # Chroma will embed this for you
        n_results=10 # how many results to return
    )

    titles = []
    asins = []
    for title in results["documents"][0]:
        if title in title_to_asin:
            asin = title_to_asin[title]
            if asin in entire_json:
                asins.append(asin)

    print("*********************")
    print("Query: " + val)
    print("")
    count = 0
    en_triples = dict()
    en_rag = dict()
    en_vector = dict()

    for en in asins:
        q = val
        count+=1
        triples_str_keywords = ""
        if en in query_engine_keywords:
            response = query_engine_keywords[en].query(q)
            triples_str_keywords = str(response)
        en_triples[en] = triples_str_keywords

        triples_str_vectors = ""
        if en in query_engine_vectors:
            response = query_engine_vectors[en].query(q)
            triples_str_vectors = str(response)

        en_vector[en] = triples_str_vectors

    docs_rag = db_chroma.similarity_search(
        query_texts=[val], # Chroma will embed this for you
        n_results=5 # how many results to return
    )

    rag_text = ""
    for rag_doc in docs_rag:
        rag_text = rag_text + "\n\n" + rag_doc.page_content

    triples_text = ""
    rag_text = ""
    vector_text = ""
    for key in en_vector.keys():
        triples = en_triples[key]
        vec = en_vector[key]
        rag_text = rag_text + "\n\n" + rag_text_en
        triples_en = f"""The following knowledge graph triples are only for the entity {key}.\n
        {triples}
        """
        triples_text = triples_text + "\n\n" + triples_en

        vector_en = f"""The following texts, extracted from a vector store, are only for the entity {key}.\n
        {vec}
        """
        vector_text = vector_text + "\n\n" + vector_en

    query_prompt = get_query_prompt(val, triples_text, rag_text, vector_text)
    system_prompt = get_query_system_prompt()
    prompt = f"""System: {system_prompt}\n\nUser: {query_prompt}\n\nAssistant:\n\n"""
    #print(prompt)
    response_text = completion(prompt, model_gpt4o)
    #display(Markdown(response_text))
    print(response_text)
    res_str = response_text
    print("*********************")

# Stream the response to the app using `st.write_stream`.
st.write(res_str)
