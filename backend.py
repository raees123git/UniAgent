"""
Backend LangGraph Workflow
Contains only the graph construction and workflow compilation
"""

from langgraph.graph import StateGraph, END
from helper_functions import (
    State,
    load_all_vector_dbs,
    user_input_node,
    query_rewriter_node,
    supervisor_node,
    create_university_agent,
    general_agent,
    quality_checker_node,
    printer_node,
    route_supervisor,
    route_quality_checker
)

# ==================== LOAD VECTOR DATABASES ====================
VECTOR_DBS = load_all_vector_dbs()

# ==================== CREATE UNIVERSITY AGENTS ====================
nust_agent = create_university_agent("NUST", VECTOR_DBS)
comsats_agent = create_university_agent("COMSATS", VECTOR_DBS)
fast_agent = create_university_agent("FAST", VECTOR_DBS)

# ==================== BUILD LANGGRAPH WORKFLOW ====================

def build_workflow():
    """Constructs and compiles the LangGraph workflow"""
    
    graph = StateGraph(State)
    
    # Add all nodes
    graph.add_node("user_input", user_input_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("nust_agent", nust_agent)
    graph.add_node("comsats_agent", comsats_agent)
    graph.add_node("fast_agent", fast_agent)
    graph.add_node("general_agent", general_agent)
    graph.add_node("quality_checker", quality_checker_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("printer", printer_node)
    
    # Set entry point
    graph.set_entry_point("user_input")
    
    # Define workflow edges
    graph.add_edge("user_input", "supervisor")
    
    # Supervisor routes to appropriate agent
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "nust_agent": "nust_agent",
            "comsats_agent": "comsats_agent",
            "fast_agent": "fast_agent",
            "general_agent": "general_agent"
        }
    )
    
    # All agents go to quality checker
    graph.add_edge("nust_agent", "quality_checker")
    graph.add_edge("comsats_agent", "quality_checker")
    graph.add_edge("fast_agent", "quality_checker")
    graph.add_edge("general_agent", "quality_checker")
    
    # Quality checker routes to printer or query rewriter
    graph.add_conditional_edges(
        "quality_checker",
        route_quality_checker,
        {
            "GOOD": "printer",
            "BAD": "query_rewriter"
        }
    )
    
    # Query rewriter loops back to supervisor
    graph.add_edge("query_rewriter", "supervisor")
    
    # Printer ends the workflow
    graph.add_edge("printer", END)
    
    # Compile and return
    return graph.compile()

# ==================== COMPILE WORKFLOW ====================
workflow = build_workflow()
print("✅ RAG workflow compiled successfully!")

# ==================== MAIN INTERFACE FUNCTION ====================

def process_query(query: str, conversation_history: list = None, university_name: str = "COMSATS"):
    """
    Main function for Streamlit to call.
    
    Args:
        query (str): User's question
        conversation_history (list): Previous conversation history
        university_name (str): Current university context (default: COMSATS)
    
    Returns:
        dict: Contains 'answer', 'university_name', and 'conversation_history'
    """
    if conversation_history is None:
        conversation_history = []
    
    input_state = {
        "user_query": query,
        "conversation_history": conversation_history,
        "university_name": university_name
    }
    
    try:
        result = workflow.invoke(input_state)
        
        return {
            "answer": result.get("answer", "Sorry, I couldn't generate an answer."),
            "university_name": result.get("university_name", university_name),
            "conversation_history": result.get("conversation_history", [])
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "answer": f"An error occurred: {str(e)}",
            "university_name": university_name,
            "conversation_history": conversation_history
        }

# ==================== CLI INTERFACE (Optional) ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Multi-University RAG System")
    print("=" * 60)
    print("\nSupported Universities: NUST, COMSATS, FAST")
    print("Type 'quit' to exit, 'clear' to reset conversation")
    print("-" * 60)
    
    # Initialize persistent state
    persistent_state = {
        "conversation_history": [],
        "university_name": "COMSATS"
    }
    
    while True:
        query = input("\n❓ Ask your question: ").strip()
        
        if query.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        if query.lower() == 'clear':
            persistent_state = {
                "conversation_history": [],
                "university_name": "COMSATS"
            }
            print("\n🔄 Conversation history cleared!")
            continue
        
        if not query:
            continue
        
        try:
            result = process_query(
                query,
                persistent_state["conversation_history"],
                persistent_state["university_name"]
            )
            
            # Update persistent state
            persistent_state["conversation_history"] = result["conversation_history"]
            persistent_state["university_name"] = result["university_name"]
            
            print("\n" + "=" * 60)
            print(f"✅ Answer (from {result['university_name']})")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")