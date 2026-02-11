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
    graph = StateGraph(State)
    
    graph.add_node("user_input", user_input_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("nust_agent", nust_agent)
    graph.add_node("comsats_agent", comsats_agent)
    graph.add_node("fast_agent", fast_agent)
    graph.add_node("general_agent", general_agent)
    graph.add_node("quality_checker", quality_checker_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("printer", printer_node)
    
    graph.set_entry_point("user_input")
    
    graph.add_edge("user_input", "supervisor")
    
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
    
    graph.add_edge("nust_agent", "quality_checker")
    graph.add_edge("comsats_agent", "quality_checker")
    graph.add_edge("fast_agent", "quality_checker")
    graph.add_edge("general_agent", "quality_checker")
    
    graph.add_conditional_edges(
        "quality_checker",
        route_quality_checker,
        {
            "GOOD": "printer",
            "BAD": "query_rewriter"
        }
    )
    
    graph.add_edge("query_rewriter", "supervisor")
    
    graph.add_edge("printer", END)
    
    return graph.compile()

# ==================== COMPILE WORKFLOW ====================
workflow = build_workflow()
print("✅ RAG workflow compiled successfully!")

# ==================== MAIN INTERFACE FUNCTION ====================

def process_query(query: str, conversation_history: list = None, university_name: str = "COMSATS"):

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


