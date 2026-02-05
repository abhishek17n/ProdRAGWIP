from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain import LLMChain
from elasticsearch import Elasticsearch
import openai
import streamlit as st
import os

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


# Elasticsearch setup
es = Elasticsearch(hosts=["http://elasticdb:9200"])

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = "62decc1207df4252b3068f828e7150b9321ce4fec3e2fbbfe823fc44ec8b8183"


if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please set it as an environment variable.")


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
        return []

# OpenAI fallback function
def search_openai(query, context=""):
    prompt = f"""
    Context: {context}
    Question: {query}
    Provide a detailed answer based on the context above. If no relevant information is found, say "I don't know."
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
    return search_openai(query, context=context)

tools = [
    Tool(name="ElasticsearchSearch", func=elasticsearch_tool, description="Search documents in Elasticsearch."),
    Tool(name="OpenAISearch", func=openai_tool, description="Search OpenAI for relevant answers if Elasticsearch has no data.")
]

# Initialize LangChain Agent
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key=openai_api_key)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Streamlit app for interaction
st.title("LangChain Agent with Elasticsearch and OpenAI")
st.write("This app uses a LangChain agent to retrieve data from Elasticsearch and OpenAI.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your query")
if st.button("Submit"):
    if query:
        # Pass the query to the agent
        response = agent.run(query)
        
        # Add query and response to the chat history
        st.session_state.chat_history.append({"query": query, "response": response})

        # Display the chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['query']}")
            st.markdown(f"**Assistant:** {chat['response']}")
            st.markdown("---")
    else:
        st.error("Please enter a query.")
