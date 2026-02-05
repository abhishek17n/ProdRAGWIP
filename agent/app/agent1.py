# Streamlit app for interaction
import streamlit as st

# Import necessary modules for agents

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools,Tool
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI

# Load environment variables from the .env file
load_dotenv()

# Consolidated API key retrieval
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")
if not serpapi_api_key:
    raise ValueError("SERPAPI_API_KEY is not set. Please set it as an environment variable.")

# Elasticsearch setup
es = Elasticsearch(hosts=["http://elasticdb:9200"])

# Elasticsearch retrieval function
def search_elasticsearch(query, top_k=5):
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["chunk", "filename", "document_id"]
            }
        },
        "size": top_k
    }
    try:
        response = es.search(index="documents", body=search_body)
        hits = response.get("hits", {}).get("hits", [])
        return [hit["_source"] for hit in hits]
    except Exception as e:
        st.error(f"Elasticsearch error: {str(e)}")
        return []

# Truncate context if too long
def truncate_context(context, max_length=2000):
    if len(context) > max_length:
        return context[:max_length] + "\n[Context truncated...]"
    return context

# OpenAI fallback function
def search_openai(query, context=""):
    prompt = f"""
    Context: {context}
    Question: {query}
    Provide a detailed answer based on the context above. If no relevant information is found, say "I don't know."
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"OpenAI error: {e}"

# Define tools for LangChain Agent
def elasticsearch_tool(query):
    results = search_elasticsearch(query)
    if results:
        return "\n".join([doc["chunk"] for doc in results])
    return "No relevant documents found in Elasticsearch."

def openai_tool(query):
    results = search_elasticsearch(query)
    context = "\n".join([doc["chunk"] for doc in results]) if results else "No context available."
    context = truncate_context(context)
    return search_openai(query, context=context)

# Initialize LangChain Agent
# tools = [
#     {
#         "name": "ElasticsearchSearch",
#         "func": elasticsearch_tool,
#         "description": "Search documents in Elasticsearch."
#     },
#     {
#         "name": "OpenAISearch",
#         "func": openai_tool,
#         "description": "Search OpenAI for relevant answers if Elasticsearch has no data."
#     }
# ]
tools = [
    Tool(name="ElasticsearchSearch", func=elasticsearch_tool, description="Search documents in Elasticsearch."),
    Tool(name="OpenAISearch", func=openai_tool, description="Search OpenAI for relevant answers if Elasticsearch has no data.")
]


# Initialize LangChain Agent
llma = ChatOpenAI(model_name="gpt-4o", temperature=0,api_key=openai_api_key)
agent = initialize_agent(tools, llma, agent="zero-shot-react-description", verbose=True)

llm = OpenAI(model_name="gpt-3.5-turbo",temperature=0.0,
            openai_api_key = openai_api_key)


# loading tools
tools = load_tools(["serpapi", 
                    "llm-math"], 
                    llm=llm)
serpapi_agent  = initialize_agent(tools, 
                        llm, 
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        verbose=True)

# Streamlit UI
st.title("LangChain Agent with Elasticsearch, OpenAI, and SerpAPI")
st.write("This app uses a LangChain agent to retrieve data from Elasticsearch, OpenAI, and SerpAPI.(google-search-results)")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inputs for Elasticsearch/OpenAI query and SerpAPI query
query = st.text_input("Enter your query for Elasticsearch/OpenAI")
serpapi_query = st.text_input("Enter your query for SerpAPI(google-search-results)")

if st.button("Submit"):
    if query or serpapi_query:
        responses = {}

        # Process Elasticsearch/OpenAI query if provided
        if query:
            responses['Elasticsearch/OpenAI'] = agent.run(query)
            
            # Add Elasticsearch/OpenAI query and response to chat history
            st.session_state.chat_history.append({"query": query, "response": responses['Elasticsearch/OpenAI']})

        # Process SerpAPI query if provided
        if serpapi_query:
            serpapi_response = serpapi_agent.run(serpapi_query)  # Assuming you have a separate agent for SerpAPI
            responses['SerpAPI'] = serpapi_response

            # Add SerpAPI query and response to chat history
            st.session_state.chat_history.append({"query": serpapi_query, "response": serpapi_response})

        # Display the chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['query']}")
            st.markdown(f"**Assistant:** {chat['response']}")
            st.markdown("---")

    else:
        st.error("Please enter at least one query.")
