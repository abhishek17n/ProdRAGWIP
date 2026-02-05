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
llma = ChatOpenAI(model_name="gpt-4o-mini", temperature=0,api_key=openai_api_key)
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
# Initialize the Gemini model using ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from crewai import Agent, Task
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
# Create a search tool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

st.title("My Agent App")
# Example usage of tools

result = scrape_tool.run(website_url="https://google.com")
st.write(result)

gemini=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the GPT-4 model using ChatOpenAI
gpt=ChatOpenAI(model="gpt-4o-2024-08-06",
               verbose=True,
               temperature=0.5,
               openai_api_key=os.getenv("OPENAI_API_KEY"))


# Data Researcher Agent using Gemini and SerperSearch
article_researcher=Agent(
    role="Senior Researcher",
    goal='Unccover ground breaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools=[search_tool],
    llm=gemini,
    allow_delegation=True

)

# Article Writer Agent using GPT
article_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  llm=gpt,
  allow_delegation=False
)

# Research Task
research_task = Task(
    description=(
        "Conduct a thorough analysis on the given {topic}."
        "Utilize SerperSearch for any necessary online research. "
        "Summarize key findings in a detailed report."
    ),
    expected_output='A 2 paragraph report on the data analysis with key insights.',
    tools=[search_tool],
    agent=article_researcher,
)

# Writing Task
writing_task = Task(
    description=(
        "Write an insightful article based on the data analysis report. "
        "The article should be clear, engaging, and easy to understand."
    ),
    expected_output='A 2-paragraph article summarizing the data insights.',
    agent=article_writer,
)

from crewai import Crew, Process

# Form the crew and define the process

crew = Crew(
        agents=[article_researcher, article_writer],
        tasks=[research_task, writing_task],
        process=Process.sequential
)
research_inputs = {
    'topic': 'The rise in global tempratures from 2018 onwards'
}





# Streamlit UI
st.title("LangChain Agent with Elasticsearch, OpenAI, SerpAPI, and Crew AI")
st.write("This app uses a LangChain agent to retrieve data from Elasticsearch, OpenAI, and SerpAPI.(google-search-results) , and supports Crew AI for task handling.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inputs for Elasticsearch/OpenAI query and SerpAPI query
query = st.text_input("Enter your query for Elasticsearch/OpenAI")
serpapi_query = st.text_input("Enter your query for SerpAPI(google-search-results)")
crew_task = st.text_input("Enter your task for Crew AI")

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
         # Process Crew AI task if provided
        if crew_task:
            # crew_response = crew_ai_handler(crew_task)
            crew_response  = crew.kickoff(inputs=research_inputs)
            responses['CrewAI'] = crew_response 

            # Add Crew AI task and response to chat history
            st.session_state.chat_history.append({"query": crew_task, "response": crew_response})

        # Display the chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['query']}")
            st.markdown(f"**Assistant:** {chat['response']}")
            st.markdown("---")

    else:
        st.error("Please enter at least one query.")
