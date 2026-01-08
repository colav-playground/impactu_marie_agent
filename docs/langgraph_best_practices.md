# LangGraph Best Practices - Implementation Guide

## Current State vs Best Practices

### ✅ What we have (Good):
- StateGraph with shared state (MessagesState pattern)
- Multi-agent orchestration
- Progress tracking with ledger
- Memory systems (OpenSearch)
- Language detection service
- Prompt engineering service

### 🔧 What needs improvement (from LangChain docs):

## 1. Tool Calling Pattern

**Current**: Agents are hardcoded in workflow
**Best Practice**: Use `@tool` decorator and let LLM decide

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool
def calculate_metrics(entity_id: str):
    """Calculate publication metrics for an entity."""
    # MongoDB aggregation for metrics
    pass

@tool
def resolve_entity(name: str):
    """Resolve entity name to canonical ID."""
    # Entity resolution logic
    pass
```

## 2. Agent with Tool Registry

**Current**: Manual orchestration between agents
**Best Practice**: Single agent with tool access

```python
from langchain.agents import create_agent

tools = [retrieve_context, calculate_metrics, resolve_entity]
prompt = (
    "You are MARIE, a research intelligence assistant. "
    "Use available tools to answer queries about scientific publications."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

## 3. StateGraph with Conditional Edges

**Current**: Linear flow with some conditions
**Best Practice**: Dynamic routing based on state

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def should_continue(state: MessagesState):
    """Decide if agent should continue or stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # Continue to execute tools
    return END  # Stop and return answer

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
```

## 4. Streaming Responses

**Current**: Batch processing
**Best Practice**: Stream for responsiveness

```python
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

## 5. Human-in-the-Loop

**Current**: Automatic execution
**Best Practice**: Add interrupts for human review

```python
from langgraph.checkpoint import MemorySaver

# Add checkpointer for interrupts
memory = MemorySaver()
graph = graph.compile(checkpointer=memory, interrupt_before=["tools"])

# Resume after human approval
graph.invoke(state, config={"configurable": {"thread_id": "123"}})
```

## 6. Durable Execution

**Current**: In-memory state
**Best Practice**: Persistent checkpoints

```python
from langgraph.checkpoint.postgres import PostgresSaver

# Use persistent checkpointer
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
graph = graph.compile(checkpointer=checkpointer)
```

## 7. Tool Response Format

**Current**: Simple string returns
**Best Practice**: Use artifacts for metadata

```python
@tool(response_format="content_and_artifact")
def retrieve_docs(query: str):
    """Returns both formatted string and raw documents."""
    docs = vector_store.similarity_search(query)
    formatted = format_docs(docs)
    return formatted, docs  # String for LLM, docs as artifact
```

## 8. Multi-Query Planning

**Current**: Single plan execution
**Best Practice**: Iterative queries

```python
# Agent automatically plans multiple tool calls:
# 1. resolve_entity("Universidad de Antioquia")
# 2. retrieve_context(entity_id, years=5)
# 3. calculate_metrics(entity_id)
# 4. rank_researchers(entity_id, limit=3)
```

## 9. Error Handling and Retries

**Best Practice**: Graceful degradation

```python
from langchain.agents.middleware import retry_with_exponential_backoff

@retry_with_exponential_backoff(max_retries=3)
def call_tool(tool, args):
    return tool.invoke(args)
```

## 10. Memory Types

**Current**: Custom OpenSearch memory
**Best Practice**: Use LangChain memory abstractions

```python
from langchain.memory import ConversationBufferMemory

# Short-term memory
short_term = ConversationBufferMemory()

# Long-term memory (semantic)
from langchain.memory import VectorStoreRetrieverMemory
long_term = VectorStoreRetrieverMemory(
    retriever=vector_store.as_retriever()
)
```

## Implementation Priority

### Phase 1 (Quick wins):
1. Convert agents to tools with `@tool` decorator
2. Add streaming responses
3. Implement proper tool response format with artifacts

### Phase 2 (Architecture):
4. Refactor to single agent + tool registry
5. Add conditional edges for dynamic routing
6. Implement proper state management

### Phase 3 (Production):
7. Add persistent checkpointing
8. Implement human-in-the-loop
9. Add comprehensive error handling
10. Deploy with LangSmith monitoring

## Example: Complete Refactored Flow

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.agents import create_agent
from langchain.tools import tool

# Define tools
@tool
def search_papers(entity: str, years: int):
    """Search papers for entity in time range."""
    pass

@tool  
def get_metrics(entity: str):
    """Get publication metrics for entity."""
    pass

# Create agent with tools
tools = [search_papers, get_metrics]
agent = create_agent(model, tools)

# Build graph
def agent_node(state):
    return {"messages": agent.invoke(state)}

def tools_node(state):
    # Execute tool calls from last message
    pass

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# Compile and run
app = graph.compile()
response = app.invoke({
    "messages": [{"role": "user", "content": query}]
})
```

## References
- [LangChain RAG Tutorial](https://docs.langchain.com/oss/python/langchain/rag)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Tool Calling Pattern](https://docs.langchain.com/oss/python/langchain/tools)
- [StateGraph Documentation](https://langchain-ai.github.io/langgraph/)
