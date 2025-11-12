"""
helper_functions.py
Helper Functions for Multi-University RAG System
Contains all utility functions, RAG chains, and LangGraph nodes
"""

from typing_extensions import TypedDict
from typing import Annotated, List, Dict
from operator import add
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== STATE DEFINITION ====================
class State(TypedDict):
    user_query: str
    university_name: str
    rewritten_query: str
    context: str
    answer: str
    quality_passed: bool
    conversation_history: Annotated[List[Dict[str, str]], add]

# ==================== CONFIGURATION ====================
VECTOR_DB_BASE_DIR = "./VectorDBs"

# ==================== VECTOR DB LOADING ====================
def load_vector_db(university_name):
    """Load FAISS vector database for a specific university"""
    vector_db_path = os.path.join(VECTOR_DB_BASE_DIR, f"{university_name}_faiss")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    return db

def load_all_vector_dbs():
    """Load all vector databases at startup"""
    print("🔄 Loading vector databases...")
    dbs = {
        "NUST": load_vector_db("nust"),
        "COMSATS": load_vector_db("comsats"),
        "FAST": load_vector_db("fast")
    }
    print("✅ All vector databases loaded!")
    return dbs

# ==================== RAG CHAIN CREATION ====================
def create_rag_chain(db, university_name):
    """Create a RAG chain for a specific university"""
    retriever = db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    uni_display_name = university_name.upper()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful university information assistant for {uni_display_name}. 
        Your job is to answer student questions using the context provided from university documents.

        Guidelines:
        - Answer directly and comprehensively using the context
        - If information is in the context, provide a complete answer
        - Don't repeat information unnecessarily
        - If the context lacks sufficient information, clearly state what's missing
        - Be friendly and professional
        - You are specifically helping with {uni_display_name}, not other universities"""),
        ("user", 
        """Context from university documents:{context}
        Question: {question}

        Answer:""")
    ])
    
    parser = StrOutputParser()
    
    def format_docs(docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([f"[Document {i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
    
    def run_chain(query):
        docs = retriever.invoke(query)
        context = format_docs(docs)
        messages = prompt.format_messages(context=context, question=query)
        answer = llm.invoke(messages)
        return parser.invoke(answer)
    
    return run_chain

# ==================== LANGGRAPH NODES ====================

def user_input_node(state: State):
    """Entry point for user queries"""
    return {"user_query": state["user_query"]}

def query_rewriter_node(state: State):
    """Rewrites vague queries to be more specific and searchable"""
    llm_rewriter = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )
    
    uni_display_name = state['university_name'].upper()
    
    try:
        prompt = f"""You are a query optimization expert. Rewrite the user's question to make it more effective for searching university documents.

        Context: The user is asking about {uni_display_name} university.

        Rules:
        1. Make vague questions more specific and searchable
        2. If the query is generic (like "tell me about", "info about", "what is"), expand it to ask about key aspects: history, campuses, programs, facilities, rankings, etc.
        3. If the query mentions "this university", "here", or uses pronouns, replace them with the actual university name
        4. Keep technical terms and specific questions as they are
        5. Output ONLY the rewritten question, nothing else

        Original question: {state['user_query']}

        Rewritten question:"""
        
        response = llm_rewriter.invoke(prompt)
        rewritten_query = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        rewritten_query = rewritten_query.replace('**', '').replace('*', '').strip()
        
        if not rewritten_query or len(rewritten_query) < 5:
            return {"rewritten_query": state['user_query']}
        
        return {"rewritten_query": rewritten_query}
        
    except Exception as e:
        return {"rewritten_query": state['user_query']}

def supervisor_node(state: State):
    """Routes queries to appropriate university agent based on context"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    query_for_prompt = state.get("rewritten_query") or state["user_query"]
    
    prompt = f"""You are a routing agent. Decide which university the user is talking about.
    Possible universities: NUST, COMSATS, FAST.

    Your task:
    1. If the query clearly mentions NUST, COMSATS, or FAST, respond with that university name
    2. If no university is mentioned and user intends to continue about {state['university_name']}, respond with {state['university_name']}
    3. If user mentions multiple universities, respond with GENERAL
    4. If user mentions another university (not NUST/COMSATS/FAST), respond with that university name
    5. Otherwise respond with GENERAL

    Examples:
    - "what is the fee structure of this university?" → {state['university_name']}
    - "which university has the best CS program?" → GENERAL
    - "tell me about bahria university" → BAHRIA

    User query: "{query_for_prompt}"
    
    Respond with ONLY the university name or GENERAL:"""
    
    resp = llm.invoke(prompt)
    uni = resp.content.strip().upper()
    
    # Normalize known universities
    if "NUST" in uni:
        uni = "NUST"
    elif "FAST" in uni:
        uni = "FAST"
    elif "COMSATS" in uni:
        uni = "COMSATS"
    
    return {"university_name": uni}

def create_university_agent(university_name: str, vector_dbs: dict):
    """Factory function to create university-specific agents"""
    def agent(state: State):
        db = vector_dbs[university_name.upper()]
        query = state.get("rewritten_query") or state["user_query"]
        rag_chain = create_rag_chain(db, university_name.lower())
        answer = rag_chain(query)
        
        new_entry = {
            "university": university_name.upper(),
            "question": state["user_query"],
            "answer": answer
        }
        
        return {
            "answer": answer,
            "conversation_history": [new_entry]
        }
    
    return agent

def general_agent(state: State):
    """General purpose agent without university-specific context"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    query = state.get("rewritten_query") or state["user_query"]
    
    prompt = f"""You are a knowledgeable assistant for university students. Answer the user's question directly, clearly and concisely.
    If user mentions a specific university, provide relevant information about that university.
    If user does not mention any university and intends to continue about {state['university_name']}, provide information about {state['university_name']} university.

    Question: {query}

    Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
    
    new_entry = {
        "university": "GENERAL",
        "question": state["user_query"],
        "answer": answer
    }
    
    return {
        "answer": answer,
        "conversation_history": [new_entry]
    }

def quality_checker_node(state: State):
    """Evaluates answer quality and decides if rewriting is needed"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    query = state.get("rewritten_query") or state["user_query"]
    
    prompt = f"""You are a quality evaluator. Rate the following answer based on:
1. Relevance to the question
2. Completeness
3. Clarity

Note: if the answer contains information related to query, even if minimal, it should be considered relevant and reply with YES.

Question: {query}
Answer: {state['answer']}

Respond with ONLY one word: YES or NO."""
    
    response = llm.invoke(prompt).content.strip().upper()
    
    if "YES" in response:
        return {"quality_passed": True, "rewritten_query": ""}
    else:
        return {"quality_passed": False, "rewritten_query": ""}

def printer_node(state: State):
    """Final node - passes state through"""
    return {}

# ==================== ROUTING FUNCTIONS ====================

def route_supervisor(state):
    """Route based on university name"""
    university = state.get("university_name", "").upper()
    
    if university == "NUST":
        return "nust_agent"
    elif university == "COMSATS":
        return "comsats_agent"
    elif university == "FAST":
        return "fast_agent"
    else:
        return "general_agent"

def route_quality_checker(state):
    """Route based on quality check result"""
    return "GOOD" if state.get("quality_passed") else "BAD"