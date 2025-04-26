import streamlit as st
from langchain_groq import ChatGroq 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler  # ✅ Updated import
import os
from dotenv import load_dotenv
# Optional: Load .env variables
load_dotenv()

#code is here
# Wrappers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=450)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# UI Setup
st.title("LangChain - Chat With Search")
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you today?"}
    ]

# Display messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check if API key is present
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        # LLM setup
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Llama3-8b-8192",
            streaming=True
        )

        # Tools setup
        tools = [search, arxiv, wiki]

        # Agent initialization
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        # Assistant response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
