import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
import requests
from bs4 import BeautifulSoup

# Load environment variables from the .env file
load_dotenv()

# Consolidated API key retrieval
google_api_key = os.getenv("GOOGLE_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable.")
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

# Custom web scraping function to replace ScrapeWebsiteTool
def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from common tags (p, h1-h6, div)
        text = ' '.join([elem.get_text(strip=True) for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div'])])
        return text[:2000] if text else "No content extracted from the website."
    except Exception as e:
        return f"Error scraping website: {str(e)}"

# Define tools for LangChain Agent
def elasticsearch_tool(query):
    results = search_elasticsearch(query)
    if results:
        return "\n".join([doc["chunk"] for doc in results])
    return "No relevant documents found in Elasticsearch."

# Initialize tools for LangChain
tools = [
    Tool(name="ElasticsearchSearch", func=elasticsearch_tool, description="Search documents in Elasticsearch.")
]

# Initialize LangChain Agent with Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=google_api_key,
    verbose=True
)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Initialize SerpAPI tools (replaces SerperDevTool)
serpapi_tools = load_tools(["serpapi", "llm-math"], llm=llm)
serpapi_agent = initialize_agent(serpapi_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Define custom scraping tool for CrewAI
def crew_scrape_tool(query):
    # Use SerpAPI to find a relevant URL, then scrape it
    serpapi_result = serpapi_agent.run(f"Find a relevant website for: {query}")
    # Extract the first URL from SerpAPI results (simplified)
    try:
        import re
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', serpapi_result)
        if urls:
            return scrape_website(urls[0])
        return "No relevant website found."
    except Exception as e:
        return f"Error processing search results: {str(e)}"

# Initialize the Gemini model for CrewAI
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=google_api_key
)

# Data Researcher Agent using Gemini
article_researcher = Agent(
    role="Senior Researcher",
    goal='Uncover groundbreaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools=[Tool(name="SearchAndScrape", func=crew_scrape_tool, description="Search for a relevant website and scrape its content.")],
    llm=gemini,
    allow_delegation=True
)

# Article Writer Agent using Gemini
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
    tools=[Tool(name="SearchAndScrape", func=crew_scrape_tool, description="Search for a relevant website and scrape its content.")],
    llm=gemini,
    allow_delegation=False
)

# Research Task
research_task = Task(
    description=(
        "Conduct a thorough analysis on the given {topic}. "
        "Use the SearchAndScrape tool to find and analyze relevant online content. "
        "Summarize key findings in a detailed report."
    ),
    expected_output='A 2 paragraph report on the data analysis with key insights.',
    tools=[Tool(name="SearchAndScrape", func=crew_scrape_tool, description="Search for a relevant website and scrape its content.")],
    agent=article_researcher
)

# Writing Task
writing_task = Task(
    description=(
        "Write an insightful article based on the data analysis report. "
        "The article should be clear, engaging, and easy to understand."
    ),
    expected_output='A 2-paragraph article summarizing the data insights.',
    agent=article_writer
)

# Form the crew and define the process
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

# Default research inputs
research_inputs = {
    'topic': 'The rise in global temperatures from 2018 onwards'
}

# Streamlit UI
st.title("LangChain Agent with Elasticsearch, SerpAPI, and Crew AI")
st.write("This app uses a LangChain agent to retrieve data from Elasticsearch, SerpAPI (Google search results), and supports Crew AI for task handling with web scraping.")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inputs for Elasticsearch query, SerpAPI query, and Crew AI task
query = st.text_input("Enter your query for Elasticsearch")
serpapi_query = st.text_input("Enter your query for SerpAPI (Google search results)")
crew_task = st.text_input("Enter your task for Crew AI")

if st.button("Submit"):
    if query or serpapi_query or crew_task:
        responses = {}

        # Process Elasticsearch query if provided
        if query:
            responses['Elasticsearch'] = agent.run(query)
            st.session_state.chat_history.append({"query": query, "response": responses['Elasticsearch']})

        # Process SerpAPI query if provided
        if serpapi_query:
            serpapi_response = serpapi_agent.run(serpapi_query)
            responses['SerpAPI'] = serpapi_response
            st.session_state.chat_history.append({"query": serpapi_query, "response": serpapi_response})

        # Process Crew AI task if provided
        if crew_task:
            crew_response = crew.kickoff(inputs={'topic': crew_task})
            responses['CrewAI'] = crew_response
            st.session_state.chat_history.append({"query": crew_task, "response": crew_response})

        # Display the chat history
        for chat in st.session_state.chat_history:
            st.markdown(f"**User:** {chat['query']}")
            st.markdown(f"**Assistant:** {chat['response']}")
            st.markdown("---")
    else:
        st.error("Please enter at least one query.")