from backend import process_query

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
        query = input("\n Ask your question: ").strip()
        
        if query.lower() == 'quit':
            print("\n Goodbye!")
            break
        
        if query.lower() == 'clear':
            persistent_state = {
                "conversation_history": [],
                "university_name": "COMSATS"
            }
            print("\n Conversation history cleared!")
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
            print(f" Answer (from {result['university_name']})")
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
            
        except Exception as e:
            print(f"\n Error: {str(e)}")