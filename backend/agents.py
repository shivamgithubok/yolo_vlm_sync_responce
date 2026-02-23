from config import Config
from openai import OpenAI
import sqlite3
import pandas as pd
from typing import Annotated, TypedDict, Literal, List, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

SEARCH_MODEL = "google/gemini-2.5-flash"
# EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=Config.OPENROUTER_API_KEY,
)

# def get_embedding(text: str) -> list[float]:
#     response = client.embeddings.create(
#         input=text,
#         model=EMBEDDING_MODEL, # Corrected model usage
#     )
#     return response.data[0].embedding

# --- Agent State ---
class AgentState(TypedDict):
    query: str
    sql_query: str
    data: list[dict]
    error: str
    messages: list
    attempts: int

# --- Nodes ---

def connect_db():
    # Placeholder for database connection
    # You might want to move this to a separate module or config
    try:
        db_path = Config.BASE_DIR / "backend" / "tracking_data.db"
        return sqlite3.connect(str(db_path))
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def generate_sql(state: AgentState):
    """Generates SQL query from natural language query."""
    query = state["query"]
    messages = state.get("messages", [])
    
    system_prompt = """You are a SQL expert. Convert the user's natural language query into a SQL query.
    Return ONLY the SQL query, without any markdown formatting or explanation.
    
    The database has the following schema:
    
    Table: tracking_objects
    Columns: id, track_id, class_name, yolo_original_class, verification_status, first_seen, last_seen, ai_processed_at, ai_common_name, ai_scientific_name, ai_description, ai_is_animal
    
    Use this schema to construct your queries. For "last detected", order by last_seen DESC.
    """
    
    response = client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    
    sql_query = response.choices[0].message.content.strip()
    # Clean up markdown if present
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    return {"sql_query": sql_query, "attempts": state.get("attempts", 0) + 1}

def execute_sql(state: AgentState):
    """Executes the generated SQL query."""
    sql_query = state["sql_query"]
    conn = connect_db()
    
    if not conn:
        return {"error": "Database connection failed", "data": []}
    
    try:
        df = pd.read_sql_query(sql_query, conn)
        # Handle NaN values which are not JSON compliant
        df = df.where(pd.notnull(df), None)
        data = df.to_dict(orient="records")
        conn.close()
        return {"data": data, "error": None}
    except Exception as e:
        if conn:
            conn.close()
        return {"error": str(e), "data": []}

def check_data(state: AgentState):
    """Checks if data was returned and is valid."""
    data = state.get("data", [])
    error = state.get("error")
    
    if error:
        return {"messages": [f"Error executing SQL: {error}"]}
        
    if not data:
        return {"messages": ["No data found matching the query."]}
        
    return {"messages": ["Data successfully retrieved."]}

def extract_data(state: AgentState):
    """Fallback: Uses AI to extract information or answer if SQL failed/returned nothing."""
    query = state["query"]
    error = state.get("error")
    sql_query = state.get("sql_query")
    
    prompt = f"""The user asked: "{query}"
    
    We tried to answer this with SQL:
    {sql_query}
    
    But encountered an error or got no results:
    {error if error else "No results found"}
    
    Please try to answer the user's question directly using your general knowledge, 
    or explain why the data might be missing based on the query.
    """
    
    response = client.chat.completions.create(
        model=SEARCH_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    answer = response.choices[0].message.content
    return {"messages": [answer]}

# --- Graph Construction ---

def should_continue(state: AgentState):
    data = state.get("data")
    error = state.get("error")
    
    if error or not data:
        return "extract_data"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("check_data", check_data)
workflow.add_node("extract_data", extract_data)

workflow.set_entry_point("generate_sql")

workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "check_data")

workflow.add_conditional_edges(
    "check_data",
    should_continue,
    {
        "extract_data": "extract_data",
        END: END
    }
)

workflow.add_edge("extract_data", END)

app = workflow.compile()

if __name__ == "__main__":
    # Test run
    initial_state = {"query": "Select all users", "messages": [], "attempts": 0}
    result = app.invoke(initial_state)
    print(result)