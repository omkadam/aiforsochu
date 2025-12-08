import streamlit as st
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SochuTV SEL Scoring"
groq_api_key = os.environ["GROQ_API_KEY"]

# -------------------------------------------------------
# 1️⃣ UPDATED PROMPT (for SEL scoring)
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an SEL Scoring Engine based on the Indian Social Emotional Learning Framework (ISELF).
Your job:

1. Ask the user exactly **5 SEL questions**, one at a time.
2. Collect the user's answers.
3. After receiving all 5 answers, generate SEL scores:
    - Self-Awareness (0–20)
    - Self-Management (0–20)
    - Social Awareness (0–20)
    - Relationship Skills (0–20)
    - Responsible Decision Making (0–20)
4. ALSO generate:
    - totalSELScore = sum of all 5
    - A short feedback summary

Rules:
- Don't give any analysis until all 5 answers are collected.
- ALWAYS output in a friendly, non-judgmental way.
- After all answers, output results in clean text (no JSON needed for now).
""",
        ),
        ("user", "{question}")
    ]
)

# -------------------------------------------------------
# LLM FUNCTION
# -------------------------------------------------------

def generate_response_new(question, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

st.title("Sochu SEL Test – Beta 1.0")
st.caption("🧠 Answer 5 questions to calculate your SEL Score")

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 600, 300)

# -------------------------------------------------------
# SESSION STATE (to store answers)
# -------------------------------------------------------

if "step" not in st.session_state:
    st.session_state.step = 1
if "answers" not in st.session_state:
    st.session_state.answers = {}

# -------------------------------------------------------
# SEL QUESTIONS (fixed)
# -------------------------------------------------------

sel_questions = {
    1: "Think about a recent moment when you felt strong emotions (anger, sadness, excitement). What did you do to handle that emotion?",
    2: "Imagine your friend is upset because someone was rude to them. What would you do in that situation?",
    3: "If you made a mistake that affected someone else, how would you handle it?",
    4: "When you feel stressed or overwhelmed, what are some ways you calm yourself down?",
    5: "If you have to make a difficult decision that affects you and your friends, how would you choose what to do?"
}

# -------------------------------------------------------
# RENDER QUESTIONS ONE BY ONE
# -------------------------------------------------------

step = st.session_state.step

if step <= 5:
    st.subheader(f"Question {step}")
    st.write(sel_questions[step])
    
    user_input = st.text_input("Your answer:", key=f"answer_{step}")

    if st.button("Submit Answer"):
        if user_input.strip() == "":
            st.warning("Please enter an answer.")
        else:
            st.session_state.answers[step] = user_input
            st.session_state.step += 1
            st.rerun()

# -------------------------------------------------------
# WHEN ALL 5 QUESTIONS ARE ANSWERED → GET SEL SCORE
# -------------------------------------------------------

elif step == 6:
    st.success("All questions answered! Calculating your SEL score...")

    combined_input = ""
    for i in range(1, 6):
        combined_input += f"Answer {i}: {st.session_state.answers[i]}\n"

    result = generate_response_new(
        question=(
            f"These are the user's 5 SEL answers. "
            f"Please analyze and give SEL scores.\n\n{combined_input}"
        ),
        temperature=temperature,
        max_tokens=max_tokens
    )

    st.subheader("Your SEL Results:")
    st.write(result)

    if st.button("Retake Test"):
        st.session_state.step = 1
        st.session_state.answers = {}
        st.rerun()
