

import asyncio
import json
import os
from typing import Annotated, Any, Literal

from dotenv import load_dotenv

# ── LangChain / LangGraph core ────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages   # reducer that appends messages
from langgraph.prebuilt import ToolNode            # ready-made tool-execution node
from langgraph.types import Command, interrupt      # human-in-the-loop primitives
from typing_extensions import TypedDict

# ── MCP ───────────────────────────────────────────────────────────────────────
from fastmcp import FastMCP                                   # server side
from langchain_mcp_adapters.client import MultiServerMCPClient  # client side

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — MCP SERVER  (runs as a separate process in production)
# ─────────────────────────────────────────────────────────────────────────────
"""
The MCP server exposes Python functions as "tools" that any MCP-compatible
client (like LangGraph) can discover and call over HTTP/SSE.

Think of it as a micro-service:
    LangGraph agent → HTTP POST → MCP Server → runs Python → returns result
"""

mcp_server = FastMCP(
    name="TaskServer",
    description="Manages users, tasks, and notifications for the AI agent",
)

# ── Fake in-memory database ───────────────────────────────────────────────────
_DB: dict[str, Any] = {
    "users": {
        "ali":   {"name": "Ali",   "email": "ali@example.com",   "role": "admin"},
        "sara":  {"name": "Sara",  "email": "sara@example.com",  "role": "user"},
        "ahmed": {"name": "Ahmed", "email": "ahmed@example.com", "role": "user"},
    },
    "tasks": {
        "ali": [
            {"id": 1, "title": "Review PRs",          "status": "pending",  "priority": "high"},
            {"id": 2, "title": "Write unit tests",    "status": "pending",  "priority": "medium"},
            {"id": 3, "title": "Update documentation","status": "done",     "priority": "low"},
            {"id": 4, "title": "Deploy to staging",   "status": "pending",  "priority": "high"},
            {"id": 5, "title": "Team standup",        "status": "pending",  "priority": "medium"},
        ],
        "sara": [
            {"id": 6, "title": "Design mockups",      "status": "pending",  "priority": "high"},
            {"id": 7, "title": "Client meeting",      "status": "done",     "priority": "high"},
        ],
        "ahmed": [
            {"id": 8, "title": "Database migration",  "status": "pending",  "priority": "critical"},
        ],
    },
}


# Each @mcp_server.tool() decorated function becomes a discoverable MCP tool.
# The docstring becomes the tool description the LLM reads to decide when to call it.

@mcp_server.tool()
def get_user_info(username: str) -> dict:
    """
    Retrieve profile information for a user.

    Args:
        username: The user's login name (case-insensitive).

    Returns:
        A dict with keys: name, email, role — or an error message.
    """
    user = _DB["users"].get(username.lower())
    if not user:
        return {"error": f"User '{username}' not found. Available: {list(_DB['users'].keys())}"}
    return user


@mcp_server.tool()
def get_pending_tasks(username: str) -> dict:
    """
    Return all PENDING tasks for a specific user, sorted by priority.

    Priority order: critical > high > medium > low

    Args:
        username: The user's login name.

    Returns:
        Dict with 'count' and 'tasks' list.
    """
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_tasks = _DB["tasks"].get(username.lower(), [])
    pending = [t for t in all_tasks if t["status"] == "pending"]
    pending.sort(key=lambda t: priority_order.get(t["priority"], 99))
    return {"count": len(pending), "tasks": pending}


@mcp_server.tool()
def complete_task(username: str, task_id: int) -> dict:
    """
    Mark a specific task as completed.

    Args:
        username: Owner of the task.
        task_id:  Numeric ID of the task to mark done.

    Returns:
        Success message or error.
    """
    tasks = _DB["tasks"].get(username.lower(), [])
    for task in tasks:
        if task["id"] == task_id:
            if task["status"] == "done":
                return {"message": f"Task {task_id} is already completed."}
            task["status"] = "done"
            return {"message": f"✅ Task {task_id} '{task['title']}' marked as done!"}
    return {"error": f"Task {task_id} not found for user '{username}'."}


@mcp_server.tool()
def create_task(username: str, title: str, priority: str = "medium") -> dict:
    """
    Create a new task for a user.

    Args:
        username: Target user.
        title:    Short description of the task.
        priority: One of 'critical', 'high', 'medium', 'low'. Default: 'medium'.

    Returns:
        The newly created task dict.
    """
    valid_priorities = {"critical", "high", "medium", "low"}
    if priority not in valid_priorities:
        return {"error": f"Invalid priority. Choose from: {valid_priorities}"}

    if username.lower() not in _DB["tasks"]:
        _DB["tasks"][username.lower()] = []

    existing_ids = [t["id"] for tasks in _DB["tasks"].values() for t in tasks]
    new_id = max(existing_ids, default=0) + 1
    new_task = {"id": new_id, "title": title, "status": "pending", "priority": priority}
    _DB["tasks"][username.lower()].append(new_task)
    return {"message": f"Task created with ID {new_id}", "task": new_task}


@mcp_server.tool()
def get_team_summary() -> dict:
    """
    Return a summary of all users and their pending task counts.
    Useful for manager-level overviews.
    """
    summary = {}
    for username in _DB["users"]:
        tasks = _DB["tasks"].get(username, [])
        pending = [t for t in tasks if t["status"] == "pending"]
        summary[username] = {
            "name": _DB["users"][username]["name"],
            "pending_tasks": len(pending),
            "critical": sum(1 for t in pending if t["priority"] == "critical"),
            "high":     sum(1 for t in pending if t["priority"] == "high"),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────
"""
State is the shared memory that flows through every node in the graph.

The `add_messages` reducer means: when a node returns {"messages": [new_msg]},
LangGraph APPENDS it to the existing list instead of replacing it.
This gives us automatic conversation history management.
"""

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # You can add more fields:
    current_user: str          # tracks which user the conversation is about
    iteration_count: int       # guards against infinite loops
    requires_approval: bool    # human-in-the-loop flag


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a smart task management assistant with access to a live task database.

When a user asks about tasks, users, or team status:
1. ALWAYS call the appropriate MCP tool to get REAL data — never guess.
2. After getting tool results, summarize them clearly and helpfully.
3. If a user asks to complete or create a task, confirm before doing it.

Available tools:
- get_user_info(username)          → user profile
- get_pending_tasks(username)      → pending tasks sorted by priority
- complete_task(username, task_id) → mark task done
- create_task(username, title, priority) → add new task
- get_team_summary()               → all users overview
"""


def build_agent_node(llm_with_tools):
    """
    Factory that returns the 'agent' node function.

    The agent node:
    1. Prepends the system prompt
    2. Calls the LLM with the full conversation history
    3. LLM returns either a plain response OR an AIMessage with tool_calls
    """
    async def agent_node(state: State) -> dict:
        print(f"\n{'='*60}")
        print(f"[AGENT NODE] Iteration #{state['iteration_count']}")
        print(f"[AGENT NODE] Last message: {state['messages'][-1].content[:100]}...")

        # Guard: prevent runaway loops
        if state["iteration_count"] > 10:
            return {
                "messages": [AIMessage(content="⚠️ Max iterations reached. Stopping.")],
                "iteration_count": state["iteration_count"],
            }

        messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response: AIMessage = await llm_with_tools.ainvoke(messages_to_send)

        print(f"[AGENT NODE] Response type: {'TOOL CALL' if response.tool_calls else 'FINAL'}")
        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"  → Calling tool: {tc['name']}({tc['args']})")

        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
        }

    return agent_node


def human_approval_node(state: State) -> Command:
    """
    HUMAN-IN-THE-LOOP node.

    This node intercepts WRITE operations (complete_task, create_task)
    and pauses the graph using `interrupt()`.

    The graph resumes only when `.update_state()` is called externally
    with the human's approval decision.

    In a web app: this is where you'd send a "Please approve?" notification.
    """
    last_msg = state["messages"][-1]

    # Find any write tool calls in the last AI message
    write_tools = {"complete_task", "create_task"}
    tool_calls = getattr(last_msg, "tool_calls", [])
    write_calls = [tc for tc in tool_calls if tc["name"] in write_tools]

    if write_calls:
        print("\n[HUMAN APPROVAL] ⚠️  Write operation detected!")
        for tc in write_calls:
            print(f"  Tool: {tc['name']}")
            print(f"  Args: {json.dumps(tc['args'], indent=4)}")

        # interrupt() PAUSES the graph here and returns control to the caller.
        # The value passed to interrupt() is shown to the human.
        decision = interrupt({
            "question": "Do you approve this action?",
            "tool_calls": write_calls,
        })

        if decision.get("approved"):
            print("[HUMAN APPROVAL] ✅ Approved — continuing to tool execution.")
            return Command(goto="tools")
        else:
            # Inject a cancellation message and skip tool execution
            print("[HUMAN APPROVAL] ❌ Rejected — cancelling.")
            cancel_msg = ToolMessage(
                content="Action cancelled by user.",
                tool_call_id=write_calls[0]["id"],
            )
            return Command(
                goto="agent",
                update={"messages": [cancel_msg]},
            )

    # No write tools → skip approval
    return Command(goto="tools")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CONDITIONAL EDGE (Router)
# ─────────────────────────────────────────────────────────────────────────────

def should_continue(state: State) -> Literal["human_approval", "tools", "__end__"]:
    """
    Router function — decides what happens after the agent node runs.

    Returns:
        "human_approval" → if the LLM wants to call a write tool
        "tools"          → if the LLM wants to call a read-only tool
        "__end__"        → if the LLM produced a final text response
    """
    last_message = state["messages"][-1]

    # No tool calls → we're done
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return END

    # Check if any tool call is a write operation
    write_tools = {"complete_task", "create_task"}
    for tc in last_message.tool_calls:
        if tc["name"] in write_tools:
            return "human_approval"   # needs approval first

    return "tools"   # safe read-only tools, go straight to execution


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

async def build_graph():
    """
    Assembles the full LangGraph state machine.

    Graph topology:
                         ┌─────────────────────┐
                         │                     │
        START ──► agent ─┤── has write tools? ──► human_approval ──► tools ──┐
                         │                     │                             │
                         └── has read tools? ──► tools ──────────────────────┤
                         │                     │                             │
                         └── no tools? ────────► END                        │
                                                                             │
                         ◄────────────────── loop back ◄────────────────────┘
    """
    # ── 1. Connect to MCP server and get tools ────────────────────────────────
    print("[SETUP] Connecting to MCP server...")
    mcp_client = MultiServerMCPClient({
        "task_server": {
            "url": "http://localhost:8000/sse",   # where FastMCP listens
            "transport": "sse",
        }
    })

    # This returns LangChain-compatible tool objects
    mcp_tools = await mcp_client.get_tools()
    print(f"[SETUP] Loaded {len(mcp_tools)} tools from MCP: {[t.name for t in mcp_tools]}")

    # ── 2. Bind tools to LLM ──────────────────────────────────────────────────
    llm = ChatAnthropic(
        model="claude-opus-4-5",
        temperature=0,          # deterministic tool selection
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(mcp_tools)

    # ── 3. Build graph ────────────────────────────────────────────────────────
    tool_node = ToolNode(mcp_tools)   # pre-built node that executes tool calls

    graph = StateGraph(State)

    # Add nodes
    graph.add_node("agent",          build_agent_node(llm_with_tools))
    graph.add_node("tools",          tool_node)
    graph.add_node("human_approval", human_approval_node)

    # Add edges
    graph.add_edge(START, "agent")

    # Conditional edge: after agent runs, where do we go?
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "human_approval": "human_approval",
            "tools":          "tools",
            END:              END,
        },
    )

    # After tools execute, always loop back to agent (it may call more tools)
    graph.add_edge("tools", "agent")

    # ── 4. Add memory (checkpointing) ─────────────────────────────────────────
    # MemorySaver stores state in RAM.
    # In production, use: SqliteSaver or PostgresSaver
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    print("[SETUP] Graph compiled successfully!")
    return compiled, mcp_client


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — STREAMING RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_conversation(graph, thread_id: str, user_input: str):
    """
    Send one message to the graph and stream the response.

    thread_id: Unique ID per conversation — enables memory across turns.
               Use the same thread_id for a multi-turn conversation.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    initial_state = {
        "messages":        [HumanMessage(content=user_input)],
        "current_user":    "ali",       # default user context
        "iteration_count": 0,
        "requires_approval": False,
    }

    print(f"\n{'─'*60}")
    print(f"USER: {user_input}")
    print(f"{'─'*60}")

    # stream_mode="values" → yields full state after each node
    # Alternatively: stream_mode="updates" → yields only changes
    final_response = None
    async for event in graph.astream(initial_state, config, stream_mode="values"):
        last_msg = event["messages"][-1]

        # Only print final AI text responses (not intermediate tool calls)
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls and last_msg.content:
            final_response = last_msg.content
            print(f"\n🤖 ASSISTANT:\n{final_response}")

        # Show tool results as they come in (optional debug output)
        elif isinstance(last_msg, ToolMessage):
            print(f"\n🔧 TOOL RESULT [{last_msg.name if hasattr(last_msg, 'name') else 'tool'}]:")
            print(f"   {last_msg.content[:200]}...")

    return final_response


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MCP SERVER RUNNER (background task)
# ─────────────────────────────────────────────────────────────────────────────

async def start_mcp_server():
    """
    Starts the MCP server on port 8000 in the background.

    In production: run this as a separate service / Docker container.
    The LangGraph agent connects to it over HTTP.
    """
    import uvicorn
    config = uvicorn.Config(
        mcp_server.get_asgi_app(),
        host="127.0.0.1",
        port=8000,
        log_level="warning",   # suppress uvicorn noise
    )
    server = uvicorn.Server(config)
    print("[MCP SERVER] Starting on http://127.0.0.1:8000/sse ...")
    await server.serve()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """
    Demonstrates:
      1. Multi-turn memory (same thread_id across calls)
      2. Tool chaining (agent calls multiple tools per query)
      3. Different users in same session
    """

    # ── Start MCP server as background task ───────────────────────────────────
    server_task = asyncio.create_task(start_mcp_server())
    await asyncio.sleep(1.5)   # wait for server to be ready

    # ── Build LangGraph ───────────────────────────────────────────────────────
    graph, mcp_client = await build_graph()

    # ── Demo conversations ────────────────────────────────────────────────────
    THREAD_A = "conversation-001"   # separate memory per thread

    # Turn 1: Read operation
    await run_conversation(
        graph,
        THREAD_A,
        "What are Ali's pending tasks? Show me the high priority ones first."
    )

    # Turn 2: Follows up on previous turn (memory works!)
    await run_conversation(
        graph,
        THREAD_A,
        "How many of those are critical priority?"
    )

    # Turn 3: Cross-user query (tool chaining — agent calls 2+ tools)
    await run_conversation(
        graph,
        THREAD_A,
        "Give me a full team summary and tell me which user has the most urgent work."
    )

    # Turn 4: Write operation — will trigger human_approval node
    # In this demo we auto-approve; in real use, you'd call graph.update_state()
    await run_conversation(
        graph,
        THREAD_A,
        "Create a new high-priority task for Ali: 'Fix production bug #42'"
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    await mcp_client.__aexit__(None, None, None)
    server_task.cancel()
    print("\n✅ Done!")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — HUMAN-IN-THE-LOOP: HOW TO RESUME FROM OUTSIDE
# ─────────────────────────────────────────────────────────────────────────────
"""
When the graph hits `interrupt()` in human_approval_node, it PAUSES.
Here's how you resume it (e.g., from a web endpoint):

    # Get the interrupted state
    state = graph.get_state(config)
    print(state.next)        # → ('human_approval',)
    print(state.tasks)       # → shows the interrupt payload

    # Resume with approval
    graph.update_state(
        config,
        {"messages": []},
        as_node="human_approval"
    )
    # Then call graph.astream(...) with None as input to continue:
    async for event in graph.astream(None, config):
        ...

This pattern enables async human review workflows where the agent
pauses, waits hours for a human, then continues.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ALTERNATIVE: LOCAL TOOLS (no MCP server needed)
# ─────────────────────────────────────────────────────────────────────────────
"""
If you don't want to run an MCP server, define tools directly with @tool:

    from langchain_core.tools import tool

    @tool
    def get_pending_tasks(username: str) -> str:
        '''Return pending tasks for user.'''
        tasks = _DB["tasks"].get(username.lower(), [])
        return json.dumps([t for t in tasks if t["status"] == "pending"])

Then pass them directly:
    llm_with_tools = llm.bind_tools([get_pending_tasks, ...])
    tool_node = ToolNode([get_pending_tasks, ...])

Everything else (State, graph, edges) stays identical.
"""


if __name__ == "__main__":
    # Verify API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ERROR: Set ANTHROPIC_API_KEY in .env file")
        print("   Create .env with: ANTHROPIC_API_KEY=sk-ant-...")
        exit(1)

    asyncio.run(main())