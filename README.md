# Agentic 75 Hard Day

This repository contains daily practice projects from the Agentic 75 Hard challenge. The work focuses on Python, LangGraph, LangChain, FastAPI, MCP tools, chatbot workflows, and agentic AI patterns.

## Contents

- `Day01` - Basic university chatbot workflow
- `Day02` - Rule-based cricket workflow
- `Day03` - Writing evaluator agent
- `Day04` - Conditional workflows and review handling
- `Day05` - AI research agent
- `Day06` - Iterative LangGraph and post generation workflows
- `Day07` - Chatbot with frontend, backend, and memory concepts
- `Day08_Langsmith` - LangSmith tracing and LCEL chains
- `Day09_Mcp` - MCP tool integration examples

## Setup

Create and activate a local Python environment:

```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```

Install the dependencies needed for the day you are working on. Some notebooks and scripts use LangChain, LangGraph, FastAPI, OpenAI, Tavily, and LangSmith.

## Environment Files

Do not commit real API keys. Use the example files as templates:

```text
.env.example
```

Copy `.env.example` to `.env` and fill in your local keys.

## Git Notes

Local virtual environments, `.env` files, Python cache files, and notebook checkpoints are ignored through `.gitignore`.

## Status

The repository preserves the daily commit history for the challenge and is ready for continued work.
