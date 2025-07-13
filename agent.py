import os
import logging
import numpy as np
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any


# --- Imports for Redis History ---
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate # Import ChatPromptTemplate


# --- Existing Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

TOP_K_RETRIEVAL = 8
BM25_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
LLM_MODEL = "llama3-70b-8192"
LLM_TEMPERATURE = 0.3

# --- Resume Data (omitted for brevity, same as before) ---
resume_json ={
    
  "personal_info": {
    "name": "Mithun MS",
    "title": "Artificial Intelligence and Machine Learning Engineer",
    "location": "Porur, Chennai",
    "email": "mithun2004vgs@gmail.com",
    "phone": "+91 8637670755",
    "linkedin": "https://www.linkedin.com/in/msmithun/",
    "github": "https://github.com/Mithun103",
    "website": "https://mithunms.netlify.app",
    "leetcode": "https://leetcode.com/u/mithun103/"
  },
  "summary": "AI/ML Engineer passionate about building smart, scalable systems and understanding the 'why' behind the code. Skilled in Python, PyTorch, TensorFlow, LLMs, and LangChain with hands-on experience in Transformer architectures, attention mechanisms, RAG pipelines, and multilingual OCR systems. Strong intuition-driven approach to solving real-world problems using GenAI, neural networks, and agent-based architectures.",
  "skills": {
    "technical": {
      "languages": ["Python", "C", "Java"],
      "frameworks": ["TensorFlow", "PyTorch", "Scikit-learn", "HuggingFace Transformers", "LangChain", "Flask", "React.js", "Gradio", "n8n"],
      "ai_ml": [
        "Deep Learning", "Neural Networks", "Transformer Architecture", "LLMs", "LoRA Fine-Tuning", "Natural Language Processing",
        "Attention Mechanisms", "Retrieval-Augmented Generation (RAG)", "OCR + LLM Integration", "YOLOv5"
      ],
      "data_science": [
        "Data Analysis", "EDA", "Feature Engineering", "Statistical Modeling", "Predictive Modeling", "Ensemble Methods", "Model Evaluation"
      ],
      "tools": ["VS Code", "Git", "GitHub", "Jupyter", "Postman", "Power BI", "Azure Vision", "MySQL", "MongoDB", "Pandas", "NumPy", "Matplotlib", "Seaborn"],
      "web_dev": ["HTML5", "CSS3", "Flask", "React.js", "RESTful APIs"],
      "others": ["API Development", "Model Deployment", "Data Preprocessing", "Competitive Programming"]
    },
    "soft_skills": ["Creativity", "Critical Thinking", "Leadership", "Problem Solving", "Collaboration", "Time Management"]
  },
  "education": [
    {
      "degree": "B.Tech in AI & ML",
      "institution": "Saveetha Engineering College, Chennai",
      "year": "2022–2026",
      "score": "CGPA: 7.5/10",
      "status": "Currently pursuing"
    },
    {
      "degree": "HSC (12th Grade)",
      "institution": "Vidhya Giri HR. Sec. School, Karaikudi",
      "year": "2022",
      "score": "89.5%"
    }
  ],
  "experience": [
    {
      "position": "AIML Engineer",
      "company": "Winvinaya Infosystem, Bangalore",
      "duration": "July 2024",
      "description": [
        "Automated spell correction in Excel using LLMs, reducing manual effort by 80%.",
        "Built multilingual OCR pipeline using Azure OCR and LLMs for contextual text correction.",
        "Developed layout-aware OCR for varied formats.",
        "Explored fine-tuning Transformers for HTML layout classification to assist VI-accessible conversion."
      ]
    },
    {
      "position": "AI/ML Intern",
      "company": "AI and ML InternPe",
      "duration": "Apr 2024 – May 2024",
      "description": [
        "Worked on AI/ML projects using Python and Scikit-learn.",
        "Handled data preprocessing, feature engineering, and data visualization."
      ]
    }
  ],
  "projects": [
    {
      "title": "Medolla: AI Medical Chatbot",
      "description": "Built a GenAI chatbot combining YOLOv5 for pathology detection, OCR for report parsing, and RAG via LangChain for interactive medical Q&A."
    },
    {
      "title": "Verba Vision Pro (OCR + LLM)",
      "description": "Integrated Azure OCR with LLMs for multilingual document parsing and contextual spell correction, supporting scanned tables and formats."
    },
    {
      "title": "T5 Summarizer with LoRA (PEFT)",
      "description": "Fine-tuned T5 using LoRA for summarization, evaluated via ROUGE metrics, and deployed with Gradio."
    },
    {
      "title": "Transformers from Scratch",
      "description": "Implemented Transformer architecture in PyTorch including multi-head attention, masking, and training loop from scratch."
    },
    {
      "title": "Temperature Prediction Pipeline",
      "description": "Built an end-to-end ML pipeline using LightGBM, XGBoost, and AdaBoost, incorporating feature selection, tuning, and evaluation."
    },
    {
      "title": "Power BI Dashboard for Sales Data",
      "description": "Designed an interactive Power BI dashboard for e-commerce sales insights and statistical business analysis."
    }
  ],
  "achievements": [
    "1st Place Winner in Inter-ML College Hackathon",
    "Top 30 in Kaggle Competitions for data science performance",
    "Finalist in SRMIST Datathon",
    "Runner-Up in Inter-College Science Quiz"
  ]
}


# --- RAG Setup (omitted for brevity, same as before) ---
def process_resume_data(resume_data: Dict[str, Any]) -> List[str]:
    docs = []
    for section, content in resume_data.items():
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, list): docs.append(f"{section} - {key}: {', '.join(value)}")
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list): docs.append(f"{section} - {key} - {sub_key}: {', '.join(sub_value)}")
                else: docs.append(f"{section} - {key}: {value}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict): docs.append(f"{section}: {', '.join([f'{k}: {v}' for k, v in item.items()])}")
                else: docs.append(f"{section}: {item}")
        else: docs.append(f"{section}: {content}")
    return docs

documents = process_resume_data(resume_json)
tokenized_corpus = [doc.lower().split() for doc in documents]
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)
bm25 = BM25Okapi(tokenized_corpus)

def retrieve_documents(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    query_vector = vectorizer.transform([query])
    tfidf_scores = (tfidf_matrix @ query_vector.T).toarray().flatten()
    norm_bm25 = bm25_scores / (np.max(bm25_scores) + 1e-9)
    norm_tfidf = tfidf_scores / (np.max(tfidf_scores) + 1e-9)
    combined_scores = (BM25_WEIGHT * norm_bm25) + (TFIDF_WEIGHT * norm_tfidf)
    num_docs = len(documents)
    if num_docs == 0: return []
    ranked_indices = combined_scores.argsort()[::-1]
    top_indices = ranked_indices[:min(top_k, num_docs)]
    return [documents[i] for i in top_indices]

# --- Agent Setup with History ---
class AgentState(TypedDict):
    user_query: str
    chat_history: List[BaseMessage] # Use chat_history instead of memory
    response: str # Add a field to store the final response

personal_ai = ChatGroq(temperature=LLM_TEMPERATURE, model=LLM_MODEL, api_key=GROQ_API_KEY)

SYSTEM_PROMPT_TEMPLATE = """You are **Mithun MS's Personal AI Assistant**. Use the following conversation history and resume context to answer the user's question.

**Conversation History:**
{chat_history}

**Resume Context:**
{retrieved_context}

**User's Question:**
{user_query}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    ("human", "{user_query}")
])

def agent_node(state: AgentState) -> Dict[str, Any]:
    """The main node of the agent that processes the user query with history."""
    query = state["user_query"]
    history = state["chat_history"]

    retrieved_docs = retrieve_documents(query)
    retrieved_context = "\n- ".join(retrieved_docs)
    logger.debug(f"Query: '{query}' | Retrieved Context:\n- {retrieved_context}")

    chain = prompt | personal_ai

    response = chain.invoke({
        "chat_history": history,
        "retrieved_context": retrieved_context,
        "user_query": query
    })

    # The graph will store the final response in the state
    return {"response": response.content}

# --- Graph Compilation ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
graph = workflow.compile()

# --- Main Entry Point for Flask ---
def personal_ai_agent(query: str, chat_history: BaseChatMessageHistory) -> Dict[str, Any]:
    """
    Invokes the agent graph with stateful history from Redis.
    """
    # Wrap the graph with message history management
    agent_with_history = RunnableWithMessageHistory(
        graph,
        lambda session_id: chat_history, # Use the history object passed from Flask
        input_messages_key="user_query",
        history_messages_key="chat_history",
        output_messages_key="response" # The key in the state holding the final response
    )

    # Invoke the agent. The session_id is managed by the RedisChatMessageHistory object.
    result = agent_with_history.invoke(
        {"user_query": query},
        config={"configurable": {"session_id": chat_history.session_id}}
    )

    # The result is now the direct response content from the AI
    return {"response": result}
