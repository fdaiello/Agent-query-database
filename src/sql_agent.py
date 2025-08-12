
import os
from typing import TypedDict, Annotated, List
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Load .env for API keys and DB credentials
load_dotenv()

# --- Step 1: Configure database connection ---
db_uri = os.getenv("DB_URI")
if not db_uri:
    raise ValueError("DB_URI not set in .env file")

db = SQLDatabase.from_uri(db_uri)


# Define the application state with memory (history)
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    history: List[str]


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)


# Define the prompt for generating SQL queries, including history
system_message = """Given an input question, create a syntactically correct {dialect} query to\
run to help find the answer. Unless the user specifies in his question a\
specific number of examples they wish to obtain, always limit your query to\
at most {top_k} results. You can order the results by a relevant column to\
return the most interesting examples in the database.\
Never query for all the columns from a specific table, only ask for a the\
few relevant columns given the question. \
Pay attention to use only the column names that you can see in the schema\
description. Be careful to not query for columns that do not exist. Also,\
pay attention to which column is in which table.\
Only use the following tables:{table_info}"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate(
    [
        ("system", system_message),
        MessagesPlaceholder("history"),
        ("user", user_prompt)
    ]
)


class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]


# Function to write the SQL query
def write_query(state: State) -> State:
    """Generate SQL query to fetch information."""
    # Prepare history as a list of strings for the prompt
    history = state.get("history", [])
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
            "history": history
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    # Update history
    new_history = history + [f"User: {state['question']}", f"SQL: {result['query']}"]
    return {**state, "query": result["query"], "history": new_history}


# Function to execute the SQL query
def execute_query(state: State) -> State:
    """Execute the SQL query and return the result."""
    result = db.run(state["query"])
    result_str = str(result)
    history = state.get("history", [])
    new_history = history + [f"SQL: {state['query']}", f"Result: {result_str}"]
    return {**state, "result": result_str, "history": new_history}


# Define the prompt for generating the answer, including history
answer_system_message = """You are a helpful AI assistant. Given the user's question and the SQL query result, \
provide a natural language answer. If the query result is empty, state that no information was found.\
"""
answer_user_prompt = "Question: {question}\nSQL Result: {result}"
answer_prompt_template = ChatPromptTemplate(
    [
        ("system", answer_system_message),
        MessagesPlaceholder("history"),
        ("user", answer_user_prompt)
    ]
)


# Function to generate the answer
def generate_answer(state: State) -> State:
    """Generate a natural language answer based on the question and query result."""
    history = [str(h) for h in state.get("history", [])]
    prompt = answer_prompt_template.invoke(
        {
            "question": state["question"],
            "result": state["result"],
            "history": history
        }
    )
    answer = llm.invoke(prompt).content
    new_history = history + [f"Answer: {answer}"]
    return {**state, "answer": answer, "history": new_history}


# Build the LangGraph workflow
workflow = StateGraph(State)

workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("write_query")
workflow.add_edge("write_query", "execute_query")
workflow.add_edge("execute_query", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()


# Example usage
if __name__ == "__main__":
    # Interactive loop for user queries
    print("\n💬 Ask me questions about your database! (type 'exit' or 'quit' to stop)")
    history: list[str] = []
    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        state: State = {
            "question": user_input,
            "query": "",
            "result": "",
            "answer": "",
            "history": history
        }
        result = app.invoke(state)
        print(f"\n📊 Answer: {result['answer']}")
        # Update history for next turn
        history = result["history"]