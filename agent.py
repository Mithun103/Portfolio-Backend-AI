from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from typing import TypedDict, List
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

# Define your personal data
documents = []
metadata = []
resume_json={
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

for section, details in resume_json.items():
    if isinstance(details, dict):
        for key, value in details.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            text = f"{sub_key}: {sub_value}"
                            documents.append(text)
                            metadata.append({"category": section, "key": sub_key})
            else:
                text = f"{key}: {value}"
                documents.append(text)
                metadata.append({"category": section, "key": key})
    elif isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                for sub_key, sub_value in item.items():
                    text = f"{sub_key}: {sub_value}"
                    documents.append(text)
                    metadata.append({"category": section, "key": sub_key})
            else:
                text = f"{section}: {item}"
                documents.append(text)
                metadata.append({"category": section, "key": section})

# 🔍 **Step 1: TF-IDF Vectorization**
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 🔍 **Step 2: BM25 Tokenization**
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# 🔎 **Combined Retrieval Function**
def retrieve_documents(query, top_k=100):
    # Tokenize query for BM25
    tokenized_query = query.lower().split()
    
    # Compute BM25 scores
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Compute TF-IDF scores
    query_vector = vectorizer.transform([query])
    tfidf_scores = np.array((tfidf_matrix @ query_vector.T).toarray()).flatten()
    
    # Combine Scores: Weighted Sum (You can adjust weights)
    combined_scores = (0.7 * bm25_scores) + (0.3 * tfidf_scores)
    
    # Get top K results
    ranked_indices = combined_scores.argsort()[-top_k:][::-1]
    
    return [documents[i] for i in ranked_indices], [metadata[i] for i in ranked_indices]

# Define the State


# Configuration
MEMORY_WINDOW_SIZE = 8
class AgentState(TypedDict):
    memory: ConversationBufferWindowMemory
    user_query: str

# Initialize AI with personal agent persona
personal_ai = ChatGoogleGenerativeAI(
    temperature=0.3,
    model="gemini-2.0-flash-exp",
    api_key="AIzaSyCs0xVuwDon6SC6c84VwdSv3txrkludhJo"
)

def agent_node(state: AgentState):
    memory = state["memory"]
    query = state["user_query"]
    
    # Retrieve relevant resume data
    ret_query, _ = retrieve_documents(query)
    
    # Create context-aware prompt
    system_prompt = f"""You are Mithun MS's Personal AI agent in mithun's portfolio website Use:
    -welcome the user,oly first time they enter.
    -important-Answer Very concisely-
    -if the user greets or said okayy ask if further assistance need
    - the following context: {ret_query}
    -give the important things in ** ** only,
    - Conversation history: {memory.buffer}
    -Answer the Queries detaily and concisely,by reading the Convo history very carefully dont mislead any info.
    -answer with all possible outcomes
    -act according to User's reply
    Rules:
    1. Answer strictly using provided data and use your knowledge only to capture the intention and treat them
    2. Be professional but friendly
    3. For unknown queries: "I don't have that information\""""
    
    # Generate response with memory context
    response = personal_ai.invoke([
        AIMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    # Update memory
    memory.save_context({"input": query}, {"output": response.content})
    
    return {"memory": memory}

# Build the workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()

# Enhanced interface with LangChain memory
def personal_ai_agent(query: str, memory: ConversationBufferWindowMemory = None):
    # Initialize memory if None
    if not memory:
        memory = ConversationBufferWindowMemory(k=MEMORY_WINDOW_SIZE, return_messages=True)
    
    # Run through workflow
    result = app.invoke({
        "memory": memory,
        "user_query": query
    })
    
    return {
        "response": result["memory"].chat_memory.messages[-1].content,
        "new_memory": result["memory"]
    }
def ask(question):
        
        global current_memory
        current_memory = None
        result = personal_ai_agent(question, current_memory)
        current_memory = result["new_memory"]
        return result["response"]
if __name__ == "__main__":
    # Conversation simulation

    
    
    
    ask("What internships have I done?")
    ask("What technologies did I use there?")
    ask("How does that relate to my education?")
