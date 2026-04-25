"""
Demo 1: Evaluation, Observability & Cost Awareness
Builds directly on: demo3_langgraph_agent_chatbot.py

Adds:
- Output evaluation
- Scoring
- Observability (tracing)
- Debugging
- Cost estimation
"""

import json
import re
import time
from typing import TypedDict, Annotated, Sequence, List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Set env vars from config.py.
import sys
import os

import config

# Start.
# -------------------------
# ENV + MODEL SETUP
# -------------------------
# TODO: Store the model name from the environment variable into a variable named MODEL_NAME



# TODO: Store the embedding model name from the environment variable into a variable named EMBED_NAME


llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# -------------------------
# STEP 1: Dataset
# -------------------------
# TODO: Import the documents from the common setup with an alias "docs"


# -------------------------
# STEP 2: Vector Store
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# STEP 3: TOOLS
# -------------------------
@tool
def rag_search(query: str) -> str:
    """Search company policy with filtering"""
    print(f"\n rag_search()...")
    results = vectorstore.similarity_search_with_score(query, k=2)

    if not results:
        return "NO_CONTEXT"

    # KEY LOGIC: threshold
    threshold = 1.2  # adjust for demo
    filtered = [doc.page_content for doc, score in results if score < threshold]

    if not filtered:
        return "NO_CONTEXT"

    return "\n".join(filtered)


@tool
def fallback_llm(query: str) -> str:
    """Fallback general LLM"""
    # print(f"\n fallback_llm()...")
    return llm.invoke(query).content


@tool
def calculate_reimbursement(text: str) -> str:
    """Calculate reimbursement"""
    # print(f"\n calculate_reimbursement()...")
    try:
        nums = re.findall(r"\d+", text)
        if len(nums) < 2:
            return "Calculation error"

        total = int(nums[0]) * int(nums[1])
        return f"Total reimbursement = ${total}"

    except:
        return "Calculation error"


# TODO: Define a collection of the tools in a varible called "tools".



# -------------------------
# NEW: OBSERVABILITY TRACE
# -------------------------
# TODO: Define a method named "add_trace".
# Receives the following arguments: state, step and info.
# The state has a property named "traces".
# Append the following values in the traces property of the state:
#   - the step.
#   - the info as a string.
#   - The current time. 



# -------------------------
# STEP 4: STATE (EXTENDED)
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int
    traces: List[Dict[str, Any]]   # NEW


MAX_RETRIES = 2

# -------------------------
# STEP 5: AGENT NODE
# -------------------------
def agent_node(state: AgentState):
    # TODO: Call the add_trace() function passing it:
    #   - the state object
    #   - step as "agent_start"
    #   - the content of the last message in the state's chat history.


    system_prompt = """
You are an intelligent assistant.

Rules:
1. Use rag_search first
2. If NO_CONTEXT:
   - Retry
   - Then fallback_llm
3. Use calculate_reimbursement for math
4. Do NOT guess
"""

    response = llm.bind_tools(tools).invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )

    add_trace(state, "agent_response", response.content)

    return {"messages": [response]}


# -------------------------
# STEP 6: ROUTER
# -------------------------
def should_continue(state: AgentState):
    # print(f"\n Router: should_continue()...")
    last_message = state["messages"][-1]

    # TODO: If the last message has tool_calls, return "tools".
    # Else return END. 



# -------------------------
# STEP 7: RETRY NODE
# -------------------------
def retry_node(state: AgentState):
    # print(f"\n NODE: retry_node()...")
    retry_count = state.get("retry_count", 0)

    add_trace(state, "retry", f"Attempt {retry_count + 1}")

    print(f"Retry attempt: {retry_count + 1}")
    if retry_count >= MAX_RETRIES:
        return {"retry_count": retry_count, "messages": []}

    last_query = [m for m in state["messages"] if m.type == "human"][-1].content
    new_query = last_query + " (more specific)"

    return {
        "messages": [HumanMessage(content=new_query)],
        "retry_count": retry_count + 1
    }


# -------------------------
# STEP 8: TOOL NODE
# -------------------------
tool_node = ToolNode(tools)


# -------------------------
# NEW: EVALUATION LOGIC
# -------------------------
def evaluate_response(text: str) -> Dict:
    """Simple rule-based evaluation"""
    # print(f"\n evaluate_response({text})...")
    score = 0
    issues = []

    # TODO: 
    # If the length of the text is greater than 10, increment score by 1.
    # else, append the string "Too short" to issues list.


    # TODO: 
    # If the text contains either a "$" or the word "policy", increment score by 1.
    # else, append the string "Low relevant" to issues list.


    # TODO: 
    # If the text contains the word "error", append the string "Error present" to issues list.


    print(f"\n score: {score}. issues: {issues}...")

    # TODO: Return a dictionary with the score and issues collection.
    # Keys are "score" and "issues".




# -------------------------
# NEW: EVALUATION NODE
# -------------------------
def evaluation_node(state: AgentState):
    # print(f"\n NODE: evaluation_node()...")
    last_msg = state["messages"][-1].content

    result = evaluate_response(last_msg)
    add_trace(state, "evaluation", result)

    return state


# -------------------------
# NEW: DEBUG NODE
# -------------------------
def debug_node(state: AgentState):
    print("\n--- TRACE LOG ---")
    for t in state["traces"]:
        print(f"{t['step']} → {t['info']}")
    print("-----------------\n")

    return state


# -------------------------
# NEW: COST ESTIMATION
# -------------------------
import tiktoken

# Pricing (USD per 1M tokens)
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60

def count_tokens(text: str, model: str):
    """Count tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    print(f"token_count: {token_count}")
    return token_count


def estimate_cost(input_text: str, output_text: str):
    """Estimate realistic OpenAI cost in USD"""
    # print(f"\n estimation_cost()...")

    print(f"input_text: {input_text}")
    input_tokens = count_tokens(input_text, os.getenv("MODEL_NAME"))

    print(f"output_text: {output_text}")
    output_tokens = count_tokens(output_text, os.getenv("MODEL_NAME"))

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M

    total_cost = input_cost + output_cost

    # TODO: Return a dictionary with the following keys and their respective values:
    # - input_token: the number of input tokens.
    # - output_token: the number of output tokens.
    # - input_cost_usd: the cost of the input tokens formatted in USD.
    # - output_cost_usd: the cost of the output tokens formatted in USD.
    # - total_cost_usd: the total cost of the input and oupout tokens formatted in USD.
    # - cost_per_1k_queries: the cost per 1,000 queries formatted in USD.



# Tool routing
def post_tool_router(state: AgentState):
    # print(f"\n post_tool_router...")
    last_msg = state["messages"][-1]
    retry_count = state.get("retry_count", 0)

    if "NO_CONTEXT" in str(last_msg):
        if retry_count < MAX_RETRIES:
            return "retry"
        else:
            return "agent"

    return "agent"

# -------------------------
# BUILD GRAPH (EXTENDED)
# -------------------------
# TODO: Build the graph.
# Add nodes for agent, tool, retry, evaluation and debug.
# Set the agent node as the entry point.
# Add a conditional edge that after the agent node, should_conitnue must be called.
# If should_continue returns "tools" call the tools node.
# If should_continue returns END, call the evaluate node.
# Add edge to call agent node after retry node.
# Add edge to call debug node after evaluate node.
# Add a conditional edge that after the tools node, post_tool_router must be called.
# If post_tool_router returns "retry" call the retry node.
# If post_tool_router returns "agent", call the agent node.
# Set debug node as the END node.


app = graph.compile()


# -------------------------
# DEMOS
# -------------------------
if __name__ == "__main__":
    print("\n=== Demo 1: Poor Output ===")

    result = app.invoke({
        "messages": [HumanMessage(content="???")],
        "retry_count": 0,
        "traces": []
    })

    print("Final:", result["messages"][-1].content)

    print("\n=== Demo 2: Good Query ===")

    # query = "If I spend $120 per day for 3 days, how much reimbursement?"
    query = "What is the leave policy?"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "retry_count": 0,
        "traces": []
    })

    print("Final:", result["messages"][-1].content)

    print("\n=== Demo 3: Multi-output Scoring ===")

    samples = [
        "Total reimbursement = $360",
        "maybe 100",
        "error happened"
    ]

    for s in samples:
        print("\nText:", s)
        print(evaluate_response(s))

    print("\n=== Demo 4: Cost Awareness ===")

    response = result["messages"][-1].content
    cost_info = estimate_cost(query, response)
    print("\n--- COST BREAKDOWN ---")
    print(json.dumps(cost_info, indent=2))

    print("\n=== Demo 5: Failure Debug ===")

    result = app.invoke({
        "messages": [HumanMessage(content="random nonsense")],
        "retry_count": 0,
        "traces": []
    })

    print("Retries:", result["retry_count"])
    