from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

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

model = ChatOpenAI(model="gpt-4o-mini")

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


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_generator():
        # Poora response lo LangGraph se (memory intact rehti hai)
        result = await app_graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
        full_text = result["messages"][-1].content

        # Word by word stream karo
        words = full_text.split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'done': True, 'thread_id': req.thread_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)