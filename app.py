"""
Streamlit Frontend for Multi-University RAG System
"""

import streamlit as st
from backend import process_query

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
        color: #000000;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        color: #000000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #1976D2;
    }
    .assistant-message .message-header {
        color: #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALIZE SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "university_name" not in st.session_state:
    st.session_state.university_name = "COMSATS"

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if "example_clicked" not in st.session_state:
    st.session_state.example_clicked = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🎓 University Assistant")
    st.markdown("---")
    
    # University selection
    st.subheader("🏛️ Select University Context")
    university_options = ["COMSATS", "NUST", "FAST"]
    
    # Handle case where university_name might not be in the list (e.g., BAHRIA, general queries)
    try:
        default_index = university_options.index(st.session_state.university_name)
    except ValueError:
        # If current university is not in options (e.g., BAHRIA), default to COMSATS
        default_index = 0
        if st.session_state.university_name not in university_options:
            st.info(f"ℹ️ Currently viewing: {st.session_state.university_name}")
    
    selected_university = st.selectbox(
        "Default University:",
        university_options,
        index=default_index
    )
    
    # Update university name if user explicitly selects from dropdown
    # Only update if it's one of the supported universities
    if selected_university in university_options and selected_university != st.session_state.get("last_selected_university"):
        st.session_state.university_name = selected_university
        st.session_state.last_selected_university = selected_university
    
    st.markdown("---")
    
    # Information section
    st.subheader("ℹ️ About")
    st.info("""
    This AI assistant helps you with information about:
    - **NUST** (National University of Sciences and Technology)
    - **COMSATS** (COMSATS University Islamabad)
    - **FAST** (FAST University)
    
    Ask questions about admissions, programs, fees, facilities, and more!
    """)
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Session Stats")
    st.metric("Total Questions", len(st.session_state.messages) // 2)
    st.metric("Current Context", st.session_state.university_name)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.example_clicked = False  # Reset welcome message
        st.rerun()

# ==================== MAIN CHAT INTERFACE ====================
st.title("💬 University Information Assistant")
st.markdown(f"**Current Context:** {st.session_state.university_name}")
st.markdown("---")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 You</div>
                    <div>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            university = message.get("university", "Assistant")
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-header">🤖 {university} Assistant</div>
                    <div>{message["content"]}</div>
                </div>
            """, unsafe_allow_html=True)

# ==================== CHAT INPUT ====================
# Use chat_input which automatically submits on Enter
user_input = st.chat_input(
    placeholder="Ask your question (e.g., What are the admission requirements for NUST?)"
)

# ==================== PROCESS USER INPUT ====================
if user_input:
    # Mark that user has started chatting - hide welcome
    st.session_state.example_clicked = True
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Store the query to process and mark as processing
    st.session_state.pending_query = user_input
    st.session_state.is_processing = True
    
    # Rerun to display the user message immediately
    st.rerun()

# ==================== WELCOME MESSAGE ====================
if not st.session_state.example_clicked and "pending_query" not in st.session_state:
    
    st.markdown("""
    ### 👋 Welcome! How can I help you today?
    
    I can answer questions about:
    - 📚 Academic programs and courses
    - 💰 Fee structure and scholarships
    - 📝 Admission requirements and deadlines
    - 🏢 Campus facilities and locations
    - 🎓 Faculty and departments
    - 📞 Contact information
    
    **Click on an example question to get started:**
    """)
    
    # Example questions as clickable buttons
    example_questions = [
        "What are the admission requirements for NUST?",
        "Tell me about COMSATS campuses",
        "What is the fee structure for CS at FAST?",
        "Compare engineering programs at comsats, fast and NUST."
    ]
    
    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
        with cols[idx % 2]:
            if st.button(
                question, 
                key=f"example_{idx}", 
                use_container_width=True
            ):
                # FIRST: Mark that example was clicked - this makes welcome disappear on next rerun
                st.session_state.example_clicked = True
                
                # SECOND: Store the query to process
                st.session_state.pending_query = question
                
                # THIRD: Mark as processing
                st.session_state.is_processing = True
                
                # FOURTH: Add user message to chat
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                
                # FINALLY: Rerun to refresh UI immediately
                st.rerun()

# ==================== PROCESS PENDING QUERY ====================
if "pending_query" in st.session_state:
    query_to_process = st.session_state.pending_query
    del st.session_state.pending_query
    
    # Show thinking indicator
    with st.spinner("🤔 Thinking..."):
        # Process query through backend
        result = process_query(
            query=query_to_process,
            conversation_history=st.session_state.conversation_history,
            university_name=st.session_state.university_name
        )
    
    # Update conversation history and university context
    st.session_state.conversation_history = result["conversation_history"]
    st.session_state.university_name = result["university_name"]
    
    # Add assistant response to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "university": result["university_name"]
    })
    
    # Mark processing as complete
    st.session_state.is_processing = False
    
    # Rerun to display new messages
    st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Powered by LangGraph + Google Gemini | Data from official university sources</small>
    </div>
""", unsafe_allow_html=True)