"""
Simple test script for the chatbot API endpoint.
Tests the agents.py integration without starting the full server.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import agents

def test_chatbot_query(query):
    """Test a single query through the agents workflow."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Invoke the agent workflow
    initial_state = {
        "query": query,
        "messages": [],
        "attempts": 0,
        "sql_query": "",
        "data": [],
        "error": None
    }
    
    try:
        result = agents.app.invoke(initial_state)
        
        print(f"\n✓ SQL Generated:")
        print(f"  {result.get('sql_query', 'N/A')}")
        
        data = result.get('data', [])
        error = result.get('error')
        
        if error:
            print(f"\n✗ Error: {error}")
        elif data:
            print(f"\n✓ Results: {len(data)} rows")
            # Show first 3 rows
            for i, row in enumerate(data[:3]):
                print(f"  Row {i+1}: {row}")
            if len(data) > 3:
                print(f"  ... and {len(data)-3} more rows")
        else:
            messages = result.get('messages', [])
            if messages:
                print(f"\n💬 AI Response:")
                for msg in messages:
                    print(f"  {msg}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Show me the last 5 detected animals",
        "How many tracks are verified?",
        "What animals were detected today?",
    ]
    
    print("\n" + "="*60)
    print("CHATBOT API TEST")
    print("="*60)
    
    for query in test_queries:
        test_chatbot_query(query)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
