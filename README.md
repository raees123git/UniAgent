🎓 UniAgent — Multi-University RAG Assistant

🧠 An intelligent University Information Assistant built using LangGraph, FAISS, and Google Gemini API.
It answers queries about NUST, COMSATS, and FAST universities using RAG (Retrieval-Augmented Generation) pipelines.

🚀 Overview

UniAgent is a multi-agent chatbot system that combines:

🧩 LangGraph — to orchestrate modular agents and reasoning steps

💬 Google Gemini API — for query understanding and generation

📚 Hugging Face Embeddings + FAISS — for semantic search in university documents

🖥️ Streamlit Frontend — for an interactive, chat-style web interface

It provides accurate, context-aware responses about admissions, programs, fees, facilities, campuses, and more — across multiple universities.


<img width="731" height="500" alt="image" src="https://github.com/user-attachments/assets/eca3e68f-b7c2-433f-b359-70ec4175f62a" />



🧠 Key Components

Component	Description

user_input_node	Entry point for new queries

supervisor_node	Decides which university agent to route the query to
university_agents	Handles NUST, COMSATS, FAST using FAISS + Gemini

quality_checker_node	Evaluates if answer is good or needs rewriting

query_rewriter_node	Rewrites vague queries for better retrieval

printer_node	Final node that outputs the response
