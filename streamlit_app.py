import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OpenAI"
groq_api_key = os.environ["GROQ_API_KEY"]

## Define our prompt template with conversation history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a psychologist who knows everything about child psychology and human psychology. Act as a facilitator and help users to learn social emotional learning. Don't make biased decisions and help user to learn social emotional learning through experiences. If you want then give them some tasks which they can perform at their home or with their friends and can learn social emotional learning."),
        ("placeholder", "{chat_history}"),
        ("user", "{question}")
    ]
)

## Function to generate response with conversation history
def generate_response_with_history(question, chat_history, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    
    # Convert chat history to the format expected by the prompt
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        else:
            formatted_history.append(AIMessage(content=msg["content"]))
    
    answer = chain.invoke({
        'question': question,
        'chat_history': formatted_history
    })
    return answer

## Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

## Page configuration
st.set_page_config(page_title="Sochu.AI", page_icon="🧠", layout="wide")

## Title of the app
st.title("Sochu.ai using Groq + LLaMA3")
st.caption("🧠 sochu.ai")

## Create two columns for metrics and clear button
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.metric(label="GPU-Temp", value="~32 °C", delta="2 °C")
with col2:
    st.metric(label="Messages", value=len(st.session_state.messages))
with col3:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

## Sidebar for settings
st.sidebar.title("Settings")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=600, value=300, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This chatbot uses LLaMA 3.1 via Groq to provide guidance on SEL. ~Sochu "
    
)

## Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

## Chat input
if prompt_input := st.chat_input("Ask me anything about psychology or social emotional learning..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response_with_history(
                prompt_input,
                st.session_state.messages[:-1],  # Exclude the current message
                temperature,
                max_tokens
            )
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the display
    st.rerun()

## Initial message if no chat history
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! I'm here to help you with psychology and social emotional learning. Ask me anything!")