from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver



class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatOpenAI()

def chatbot(state: ChatbotState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

checkpoint = MemorySaver()
graph = StateGraph(ChatbotState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app_graph = graph.compile(checkpointer=checkpoint)



app = FastAPI(title="LangGraph Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    reply: str
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = app_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=req.thread_id)

@app.get("/health")
async def health():
    return {"status": "ok"}

