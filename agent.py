# agent.py
import os
import logging
import numpy as np
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

MEMORY_WINDOW_SIZE = 8
TOP_K_RETRIEVAL = 8
BM25_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
LLM_MODEL = "llama3-70b-8192"
LLM_TEMPERATURE = 0.3

# --- Resume Data ---
# It's better to load this from a separate resume.json file
# For this example, we'll keep it here.
resume_json = {
    "personal_info": {
      "name": "Mithun MS",
      "title": "Artificial Intelligence and Machine Learning Engineer",
      "location": "Porur, Chennai",
      "email": "mithun2004vgs@gmail.com",
      "phone": "+91 8637670755",
      "linkedin": "www.linkedin.com/in/mithun-ms-5836b3297/",
      "github": "github.com/Mithun103",
      "website": "mithunms.netlify.app"
    },
    "summary": "AI & ML Engineer skilled in Python, TensorFlow, PyTorch, Large Language Models (LLMs), Neural Networks, and Data Science, with expertise in Agentic AI using LangChain and Phidata. Proficient in Transformer Architecture, Attention Mechanisms, and RAG (Retrieval-Augmented Generation). Experienced in building scalable AI-driven solutions to solve real-world challenges. Passionate about advancing AI technologies through innovation and collaboration.",
    "skills": {
      "technical": {
        "languages": ["Python", "C", "Java"],
        "frameworks": ["TensorFlow", "PyTorch", "Scikit-learn", "Phidata", "LangChain", "n8n", "Flask"],
        "ai_ml": ["Neural Networks", "Deep Learning", "Large Language Models (LLMs)", "Transformer Architecture", "Attention Mechanisms", "RAG", "NLP", "Agentic AI"],
        "data_science": ["Data Analysis", "Data Visualization", "Statistical Modeling", "Feature Engineering", "Predictive Modeling"],
        "tools": ["Jupyter", "Azure Vision", "Power BI", "Pandas", "NumPy", "Matplotlib", "Seaborn"],
        "others": ["Model Deployment", "Data Preprocessing", "API Development"]
      },
      "soft_skills": ["Creativity", "Critical Thinking", "Leadership", "Problem Solving", "Collaboration", "Time Management"]
    },
    "education": [
      {
        "degree": "B.Tech in AI & ML",
        "institution": "Saveetha Engineering College",
        "year": "2022–2026",
        "status": "Currently pursuing"
      },
      {
        "degree": "HSC (12th Grade)",
        "institution": "Vidhya Giri HR. Sec. School",
        "year": "2022",
        "score": "89.5%"
      }
    ],
    "certifications": [
      "Generative AI Course by GUVI",
      "Introduction to Deep Learning by Infosys SpringBoard",
      "Artificial Intelligence Bootcamp by NOVITech R&D",
      "Power BI Workshop by Tech Tips"
    ],
    "projects": [
      {
        "title": "Power BI Dashboard for Sales Data Analysis",
        "date": "June 2024",
        "description": "Designed an interactive dashboard for e-commerce sales analysis, leveraging data visualization and statistical insights to drive business decisions."
      },
      {
        "title": "Medical Chatbot (Combining Traditional and Gen AI)",
        "date": "Sept 2024",
        "description": "Developed a Flask-based chatbot with LLaMA 3.1 and RAG (Retrieval-Augmented Generation) for medical diagnoses."
      },
      {
        "title": "AI-Powered Quiz Maker App",
        "date": "Aug 2024",
        "description": "Built an AI-powered app to generate quizzes using NLP and LLMs."
      },
      {
        "title": "Temperature Prediction Model",
        "date": "Sept 2024",
        "description": "Created a stacked ensemble model (LGBM, RandomForest, AdaBoost) for weather forecasting using Neural Networks. Applied feature engineering and data preprocessing to improve model accuracy."
      },
      {
        "title": "Contextual Spell Correction using LLaMA",
        "date": "July 2024",
        "description": "Implemented spell correction for Excel datasets using LLM. Leveraged data science workflows for cleaning and transforming text data."
      }
    ],
    "experience": [
      {
        "position": "AI/ML Intern",
        "company": "AI and ML InternPe",
        "duration": "Apr 2024 – May 2024",
        "description": "Worked on AI/ML projects using Python, Scikit-learn, and data preprocessing. Applied data science techniques to analyze and visualize datasets."
      },
      {
        "position": "AIML Engineer",
        "company": "Winvinaya Infosystem, Bangalore",
        "duration": "July 2024",
        "description": "Enhanced data processing with spell correction and multi-language text extraction using advanced AI techniques like NLP and LLMs. Utilized data science methodologies for data cleaning and transformation."
      }
    ],
    "hackathon_experience": [
      "IIT Shaastra’s AIML Challenge 1 & 2",
      "SRMIST Datathon",
      "Intel oneAPI Hackathon",
      "IBM Z Datathon",
      "Industrial AI Hackathon (IIT Madras)"
    ]
  }

# --- RAG Setup ---
def process_resume_data(resume_data: Dict[str, Any]) -> List[str]:
    """Flattens the resume JSON into a list of strings for retrieval."""
    docs = []
    for section, content in resume_data.items():
        if isinstance(content, dict):
            # For personal_info and skills
            for key, value in content.items():
                if isinstance(value, list):
                    docs.append(f"{section} - {key}: {', '.join(value)}")
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                         if isinstance(sub_value, list):
                             docs.append(f"{section} - {key} - {sub_key}: {', '.join(sub_value)}")
                else:
                    docs.append(f"{section} - {key}: {value}")
        elif isinstance(content, list):
            # For projects, experience, education, etc.
            for item in content:
                if isinstance(item, dict):
                    doc_str = ', '.join([f"{k}: {v}" for k, v in item.items()])
                    docs.append(f"{section}: {doc_str}")
                else:
                    # For simple lists like certifications
                    docs.append(f"{section}: {item}")
        else:
            # For top-level fields like summary
            docs.append(f"{section}: {content}")
    return docs

documents = process_resume_data(resume_json)
tokenized_corpus = [doc.lower().split() for doc in documents]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_documents(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    """Retrieves relevant documents using a hybrid BM25 and TF-IDF approach."""
    tokenized_query = query.lower().split()
    
    # BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # TF-IDF scores
    query_vector = vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()

    # Normalize scores to be in a similar range (e.g., 0-1) before combining
    norm_bm25 = bm25_scores / (np.max(bm25_scores) + 1e-9)
    norm_tfidf = tfidf_scores / (np.max(tfidf_scores) + 1e-9)

    combined_scores = (BM25_WEIGHT * norm_bm25) + (TFIDF_WEIGHT * norm_tfidf)
    
    # Get top_k indices, handling cases where there are fewer docs than top_k
    num_docs = len(documents)
    if num_docs == 0:
        return []
    
    ranked_indices = combined_scores.argsort()[::-1]
    top_indices = ranked_indices[:min(top_k, num_docs)]
    
    return [documents[i] for i in top_indices]

# --- Agent Setup ---
class AgentState(TypedDict):
    memory: ConversationBufferWindowMemory
    user_query: str

personal_ai = ChatGroq(
    temperature=LLM_TEMPERATURE,
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
)

SYSTEM_PROMPT_TEMPLATE = """You are **Mithun MS's Personal AI Assistant**, designed to represent his portfolio in a smart and conversational way.

🎯 **Your Objective:**
- Act as a polite and professional assistant representing Mithun.
- **Strictly use only the provided resume data** to answer all questions. Do not invent or assume any information.
- Your goal is to help users, like recruiters or collaborators, learn about Mithun's skills and experience.

---
📄 **Context (Retrieved from Mithun's Resume):**
{retrieved_context}
---

🧠 **Conversation History:**
{conversation_history}
---

📌 **Interaction Rules:**
1.  **First Greeting:** Greet the user warmly on their very first message only.
2.  **Concise & Clear:** Always answer concisely, but with enough detail to be useful. Use **bold styling (with asterisks)** to highlight key terms, technologies, or achievements.
3.  **Handle Missing Info:** If the answer is not in the provided context, you MUST reply with: "I don't have that specific information in Mithun's resume."
4.  **Closing Prompt:** If the user says "ok", "thanks", "thank you", or similar signs of ending the conversation, respond with: "You're welcome! Is there anything else I can help you with regarding Mithun's profile?"
5.  **Tone:** Maintain a friendly, crisp, and helpful tone throughout the conversation.
"""

def agent_node(state: AgentState) -> Dict[str, Any]:
    """The main node of the agent that processes the user query."""
    memory = state["memory"]
    query = state["user_query"]

    retrieved_docs = retrieve_documents(query)
    retrieved_context = "\n- ".join(retrieved_docs)
    logger.debug(f"Query: '{query}' | Retrieved Context:\n- {retrieved_context}")

    convo_history = "\n".join([f"{m.type.upper()}: {m.content}" for m in memory.chat_memory.messages])

    formatted_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_context=retrieved_context,
        conversation_history=convo_history
    )

    response = personal_ai.invoke([
        AIMessage(content=formatted_prompt),
        HumanMessage(content=query)
    ])

    memory.save_context({"input": query}, {"output": response.content})
    return {"memory": memory}

# --- Graph Compilation ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
compiled_app = workflow.compile()

# --- Main Entry for Flask ---
def personal_ai_agent(query: str, memory: ConversationBufferWindowMemory) -> Dict[str, Any]:
    """Invokes the compiled agent graph and returns the response."""
    result = compiled_app.invoke({
        "memory": memory,
        "user_query": query
    })
    # The last message in memory is the AI's latest response
    response_content = result["memory"].chat_memory.messages[-1].content
    return {"response": response_content}
