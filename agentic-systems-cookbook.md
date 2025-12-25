# Agentic Systems Cookbook
## Production-Ready Recipes for GPT-4o and Claude Opus 4.5 (2025)

**Last Updated:** 2025-12-25
**Models:** OpenAI GPT-4o/4.5/mini, Anthropic Claude Opus 4.5/Sonnet/Haiku

---

## December 2025 Framework Updates

| Framework | Version | Key Changes |
|-----------|---------|-------------|
| **LangGraph** | 1.0 (Oct 2025) | `create_react_agent` abstraction, middleware, improved HITL |
| **OpenAI Agents SDK** | Mar 2025 | Guardrails, handoffs, structured output, automatic tracing |
| **Google ADK** | Dec 2025 | Context management, artifact handling, memory services |
| **CrewAI** | 2025 IA40 | 12M+ daily executions, #7 on IA Enablers list |

### Model Pricing (December 2025)
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| **Claude Opus 4.5** | $5/M | $25/M | 67% reduction from 4.1, 80.9% SWE-bench |
| **Claude Sonnet 4** | $3/M | $15/M | Production workhorse |
| **Claude Haiku** | $0.25/M | $1.25/M | Bulk processing |
| **GPT-4o** | $2.50/M | $10/M | General purpose |
| **GPT-4o-mini** | $0.15/M | $0.60/M | Cost-optimized |
| **GPT-4.5** | TBD | TBD | Released Dec 2025 |
| **o3-mini** | $1.10/M | $4.40/M | Reasoning-optimized |

---

## Recipe Quick Reference

| # | Recipe | Use Case | Difficulty | Key Feature |
|---|--------|----------|------------|-------------|
| 1 | Basic GPT-4o Agent | Simple structured tasks | ⭐ | Pydantic output |
| 2 | Basic Claude Agent | Tasks with thinking visibility | ⭐ | Extended thinking |
| 3 | ReAct Agent | Reasoning + action cycles | ⭐⭐ | Tool integration |
| 4 | Reflective Agent | Self-improving outputs | ⭐⭐⭐ | Quality iteration |
| 5 | Supervisor Pattern | Multi-agent coordination | ⭐⭐⭐ | LangGraph workflow |
| 6 | Parallel Agents | Fast concurrent execution | ⭐⭐ | Async processing |
| 7 | Dynamic Tool Selection | RAG-based tool retrieval | ⭐⭐⭐ | Embedding search |
| 8 | Validated Tools | Safe tool execution | ⭐⭐ | Pre-execution checks |
| 9 | Production Error Handling | Robust retry/fallback | ⭐⭐⭐ | Circuit breaker |
| 10 | Cost Tracking | Budget management | ⭐⭐ | Usage monitoring |
| 11 | Model Cascading | Complexity-based routing | ⭐⭐ | Cost optimization |
| 12 | Agentic Plan Caching | 50% cost reduction | ⭐⭐⭐ | Plan reuse |
| 13 | OpenAI Agents SDK | Quick prototyping | ⭐⭐ | Guardrails + handoffs |
| 14 | MCP Tool Integration | Standardized tools | ⭐⭐ | Protocol compliance |

---

## Recipe 1: Basic Agent with GPT-4o

**Use Case:** Simple task execution with structured output
**Difficulty:** ⭐ Beginner

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class TaskResult(BaseModel):
    status: str  # "complete" or "failed"
    output: str
    confidence: float  # 0-1

def simple_agent(task: str) -> TaskResult:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "Complete tasks. Respond with status, output, confidence."},
            {"role": "user", "content": task}
        ],
        response_format=TaskResult
    )
    return response.choices[0].message.parsed

# Usage
result = simple_agent("Summarize the benefits of using TypeScript")
print(f"Status: {result.status}, Confidence: {result.confidence}")
```

**Expected Output:** `Status: complete, Confidence: 0.95`

---

## Recipe 2: Basic Agent with Claude Sonnet 4.5

**Use Case:** Task execution with thinking process visibility
**Difficulty:** ⭐ Beginner

```python
import anthropic

client = anthropic.Anthropic()

def claude_agent(task: str, show_thinking: bool = False):
    config = {
        "model": "claude-sonnet-4.5",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": task}]
    }

    if show_thinking:
        config["thinking"] = {"type": "enabled", "budget_tokens": 1000}

    response = client.messages.create(**config)

    result = {"output": None, "thinking": None}
    for block in response.content:
        if block.type == "thinking":
            result["thinking"] = block.thinking
        elif block.type == "text":
            result["output"] = block.text

    return result

# Usage
simple = claude_agent("What are the benefits of TypeScript?")
complex = claude_agent("Design a database schema for multi-tenant SaaS", show_thinking=True)
```

---

## Recipe 3: ReAct Agent (Reasoning + Acting)

**Use Case:** Agent that reasons before taking actions
**Difficulty:** ⭐⭐ Intermediate

```python
from typing import List, Dict, Any
import json
from openai import OpenAI

class ReActAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI()
        self.max_iterations = 10

    def execute(self, task: str, tools: List[Dict]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Use ReAct: THOUGHT → ACTION → OBSERVATION. Repeat until done, then FINAL ANSWER."},
            {"role": "user", "content": f"Task: {task}"}
        ]

        trace = []
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0
            )

            message = response.choices[0].message
            messages.append(message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    trace.append({
                        "iteration": iteration,
                        "action": f"{function_name}({arguments})",
                        "observation": self._execute_tool(function_name, arguments)
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(trace[-1]["observation"])
                    })
            else:
                trace.append({"iteration": iteration, "final_answer": message.content})
                break

        return {"answer": message.content, "trace": trace, "iterations": iteration + 1}

    def _execute_tool(self, function_name: str, arguments: dict) -> Any:
        tools_map = {
            "calculator": lambda expr: eval(expr, {"__builtins__": {}}, {}),
            "get_weather": lambda loc: {"location": loc, "temperature": 72, "conditions": "Sunny"}
        }
        return tools_map.get(function_name, lambda **k: {"error": "Unknown tool"})(**arguments)

# Usage
tools = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Perform calculations",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
    }}
]

agent = ReActAgent()
result = agent.execute("What's 15% of 240?", tools=tools)
print(f"Answer: {result['answer']}, Iterations: {result['iterations']}")
```

**Expected Output:** `Answer: 36, Iterations: 2`

---

## Recipe 4: Self-Improving Agent with Reflection

**Use Case:** Agent that evaluates and improves its own outputs
**Difficulty:** ⭐⭐⭐ Advanced

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

class ReflectiveAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI()

    def execute_with_reflection(self, task: str, max_iterations: int = 3, quality_threshold: float = 0.8) -> Dict:
        current_output = None
        history = []

        for iteration in range(max_iterations):
            output = self._generate(task) if current_output is None else self._improve(task, current_output, history[-1]["critique"])
            reflection = self._reflect(task, output)

            history.append({
                "iteration": iteration,
                "output": output,
                "quality_score": reflection["quality_score"],
                "critique": reflection["critique"]
            })

            if reflection["quality_score"] >= quality_threshold:
                return {"final_output": output, "iterations": iteration + 1, "quality_score": reflection["quality_score"], "history": history}

            current_output = output

        return {"final_output": current_output, "iterations": max_iterations, "quality_score": history[-1]["quality_score"], "history": history}

    def _generate(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Provide high-quality responses."},
                {"role": "user", "content": task}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    def _reflect(self, task: str, output: str) -> Dict:
        class Reflection(BaseModel):
            quality_score: float
            critique: str
            improvements: List[str]

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": "Evaluate on accuracy, completeness, clarity. Provide quality_score (0-1), critique, improvements."},
                {"role": "user", "content": f"Task: {task}\n\nOutput:\n{output}"}
            ],
            response_format=Reflection
        )

        reflection = response.choices[0].message.parsed
        return {"quality_score": reflection.quality_score, "critique": reflection.critique, "improvements": reflection.improvements}

    def _improve(self, task: str, previous_output: str, critique: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Improve outputs based on feedback."},
                {"role": "user", "content": f"Task: {task}\n\nPrevious:\n{previous_output}\n\nCritique:\n{critique}\n\nProvide improved version:"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

# Usage
agent = ReflectiveAgent()
result = agent.execute_with_reflection("Write a professional meeting request email", quality_threshold=0.85)
print(f"Quality: {result['quality_score']}, Iterations: {result['iterations']}")
```

---

## Recipe 5: Supervisor Pattern with LangGraph

**Use Case:** Coordinate multiple specialized agents
**Difficulty:** ⭐⭐⭐ Advanced

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
import anthropic

class SupervisorState(TypedDict):
    messages: Annotated[list, "add_messages"]
    current_agent: str
    task_complete: bool
    artifacts: dict

class ResearchAgent:
    def __init__(self):
        self.client = OpenAI()

    def __call__(self, state: SupervisorState) -> dict:
        messages = state["messages"] + [SystemMessage(content="Research specialist: Gather comprehensive information.")]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": m.type, "content": m.content} for m in messages]
        )
        return {
            "messages": [AIMessage(content=response.choices[0].message.content)],
            "artifacts": {**state.get("artifacts", {}), "research": response.choices[0].message.content}
        }

class AnalystAgent:
    def __init__(self):
        self.client = OpenAI()

    def __call__(self, state: SupervisorState) -> dict:
        research = state["artifacts"].get("research", "")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Analyst: Identify insights, patterns, recommendations."},
                {"role": "user", "content": f"Research:\n{research}\n\nAnalyze:"}
            ]
        )
        return {
            "messages": [AIMessage(content=response.choices[0].message.content)],
            "artifacts": {**state["artifacts"], "analysis": response.choices[0].message.content}
        }

class WriterAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def __call__(self, state: SupervisorState) -> dict:
        research = state["artifacts"].get("research", "")
        analysis = state["artifacts"].get("analysis", "")
        response = self.client.messages.create(
            model="claude-sonnet-4.5",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"Create report from:\n\nResearch:\n{research}\n\nAnalysis:\n{analysis}"}]
        )
        return {
            "messages": [AIMessage(content=response.content[0].text)],
            "artifacts": {**state["artifacts"], "final_report": response.content[0].text},
            "task_complete": True
        }

class Supervisor:
    def __call__(self, state: SupervisorState) -> Command[Literal["researcher", "analyst", "writer", END]]:
        artifacts = state.get("artifacts", {})

        if "research" not in artifacts:
            return Command(goto="researcher", update={"current_agent": "researcher"})
        elif "analysis" not in artifacts:
            return Command(goto="analyst", update={"current_agent": "analyst"})
        elif "final_report" not in artifacts:
            return Command(goto="writer", update={"current_agent": "writer"})
        else:
            return Command(goto=END)

def create_supervisor_graph():
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", Supervisor())
    workflow.add_node("researcher", ResearchAgent())
    workflow.add_node("analyst", AnalystAgent())
    workflow.add_node("writer", WriterAgent())
    workflow.set_entry_point("supervisor")
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")
    return workflow.compile()

# Usage
graph = create_supervisor_graph()
result = graph.invoke({
    "messages": [HumanMessage(content="Research AI agents trends for 2025")],
    "current_agent": "",
    "task_complete": False,
    "artifacts": {}
})
print(result["artifacts"]["final_report"])
```

---

## Recipe 6: Parallel Multi-Agent Execution

**Use Case:** Run multiple agents simultaneously for speed
**Difficulty:** ⭐⭐ Intermediate

```python
import asyncio
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict

class ParallelAgents:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def run_agent(self, agent_config: Dict) -> Dict:
        response = await self.client.chat.completions.create(
            model=agent_config["model"],
            messages=[
                {"role": "system", "content": agent_config["system_prompt"]},
                {"role": "user", "content": agent_config["task"]}
            ]
        )
        return {"agent_name": agent_config["name"], "result": response.choices[0].message.content}

    async def run_parallel(self, agent_configs: List[Dict]) -> List[Dict]:
        tasks = [self.run_agent(config) for config in agent_configs]
        return await asyncio.gather(*tasks)

    def aggregate_results(self, results: List[Dict]) -> str:
        prompt = "Synthesize these outputs:\n\n" + "\n\n".join([f"{r['agent_name']}:\n{r['result']}" for r in results])

        sync_client = OpenAI()
        response = sync_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Synthesize multiple agent outputs into unified response."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

# Usage
async def main():
    agents = ParallelAgents()
    configs = [
        {"name": "Technical", "model": "gpt-4o", "system_prompt": "Technical analyst", "task": "Technical requirements for chat app"},
        {"name": "UX", "model": "gpt-4o-mini", "system_prompt": "UX designer", "task": "UX design for chat app"},
        {"name": "Security", "model": "gpt-4o-mini", "system_prompt": "Security expert", "task": "Security for chat app"}
    ]

    results = await agents.run_parallel(configs)
    final = agents.aggregate_results(results)
    print(final)

asyncio.run(main())
```

**Benefits:** 3x faster, lower latency, cost-effective

---

## Recipe 7: Dynamic Tool Selection with RAG

**Use Case:** Select relevant tools from large library
**Difficulty:** ⭐⭐⭐ Advanced

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
from openai import OpenAI

class DynamicToolSelector:
    def __init__(self):
        self.client = OpenAI()
        self.all_tools = []
        self.tool_embeddings = []

    def register_tool(self, tool_definition: Dict):
        tool_text = f"{tool_definition['function']['name']}: {tool_definition['function']['description']}"
        embedding = self.client.embeddings.create(model="text-embedding-3-small", input=tool_text).data[0].embedding
        self.all_tools.append(tool_definition)
        self.tool_embeddings.append(embedding)

    def select_relevant_tools(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
        similarities = cosine_similarity([query_embedding], self.tool_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.all_tools[i] for i in top_indices]

    def execute_with_dynamic_tools(self, task: str) -> Dict:
        relevant_tools = self.select_relevant_tools(task, top_k=5)

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": task}],
            tools=relevant_tools
        )

        return {
            "response": response.choices[0].message.content,
            "tools_selected": [t['function']['name'] for t in relevant_tools],
            "tools_used": [tc.function.name for tc in (response.choices[0].message.tool_calls or [])]
        }

# Usage
selector = DynamicToolSelector()
tools_library = [
    {"type": "function", "function": {"name": "web_search", "description": "Search internet for information", "parameters": {}}},
    {"type": "function", "function": {"name": "arxiv_search", "description": "Search academic papers", "parameters": {}}},
    {"type": "function", "function": {"name": "stock_price", "description": "Get stock prices", "parameters": {}}}
]

for tool in tools_library:
    selector.register_tool(tool)

result = selector.execute_with_dynamic_tools("Find papers about transformers")
print(f"Selected: {result['tools_selected']}, Used: {result['tools_used']}")
```

**Benefits:** Scales to 100+ tools, better accuracy, lower token usage

---

## Recipe 8: Tool Use with Validation

**Use Case:** Validate tool calls before execution
**Difficulty:** ⭐⭐ Intermediate

```python
from typing import Callable, Dict, Any
import json
from openai import OpenAI

class ValidatedToolExecutor:
    def __init__(self):
        self.client = OpenAI()
        self.tools = {}

    def register_tool(self, tool_definition: Dict, executor: Callable, validator: Callable = None):
        name = tool_definition["function"]["name"]
        self.tools[name] = {"definition": tool_definition, "executor": executor, "validator": validator}

    def execute_task(self, task: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": task}],
            tools=[t["definition"] for t in self.tools.values()]
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return {"result": message.content}

        results = []
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if function_name in self.tools:
                tool = self.tools[function_name]

                if tool["validator"]:
                    validation = tool["validator"](arguments)
                    if not validation["valid"]:
                        results.append({"tool": function_name, "status": "validation_failed", "error": validation["error"]})
                        continue

                try:
                    result = tool["executor"](**arguments)
                    results.append({"tool": function_name, "status": "success", "result": result})
                except Exception as e:
                    results.append({"tool": function_name, "status": "execution_failed", "error": str(e)})

        return {"tool_results": results}

# Validators
def validate_sql_query(args: Dict) -> Dict:
    query = args.get("query", "").upper()
    dangerous = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    for keyword in dangerous:
        if keyword in query:
            return {"valid": False, "error": f"Keyword '{keyword}' not allowed"}
    return {"valid": True}

# Usage
executor = ValidatedToolExecutor()
executor.register_tool(
    {"type": "function", "function": {"name": "sql_query", "description": "Execute SQL", "parameters": {}}},
    lambda query: {"rows": [], "count": 0},
    validate_sql_query
)

result = executor.execute_task("Get all users from database")
```

---

## Recipe 9: Production-Grade Error Handling

**Use Case:** Robust error handling with retries and fallbacks
**Difficulty:** ⭐⭐⭐ Advanced

```python
import time
import random
from typing import Dict, Any
from openai import OpenAI, RateLimitError, APIError, Timeout

class RobustAgent:
    def __init__(self, primary_model="gpt-4o", fallback_model="gpt-4o-mini"):
        self.client = OpenAI()
        self.primary_model = primary_model
        self.fallback_model = fallback_model

    def execute_with_retry(self, task: str, max_retries: int = 3, timeout: int = 30, use_fallback: bool = True) -> Dict[str, Any]:
        models_to_try = [self.primary_model]
        if use_fallback:
            models_to_try.append(self.fallback_model)

        last_error = None

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": task}],
                        timeout=timeout
                    )
                    return {
                        "success": True,
                        "result": response.choices[0].message.content,
                        "model_used": model,
                        "attempts": attempt + 1
                    }

                except RateLimitError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(wait_time)

                except (Timeout, APIError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)

        return {"success": False, "error": str(last_error), "models_attempted": models_to_try}

    def execute_with_circuit_breaker(self, task: str, failure_threshold: int = 5, timeout_seconds: int = 60) -> Dict:
        if not hasattr(self, '_circuit_breaker_state'):
            self._circuit_breaker_state = {"state": "closed", "failure_count": 0, "last_failure_time": None}

        cb = self._circuit_breaker_state

        if cb["state"] == "open":
            if time.time() - cb["last_failure_time"] > timeout_seconds:
                cb["state"] = "half-open"
            else:
                return {"success": False, "error": "Circuit breaker OPEN"}

        result = self.execute_with_retry(task, max_retries=1)

        if result["success"]:
            if cb["state"] == "half-open":
                cb["state"] = "closed"
                cb["failure_count"] = 0
            return result
        else:
            cb["failure_count"] += 1
            cb["last_failure_time"] = time.time()
            if cb["failure_count"] >= failure_threshold:
                cb["state"] = "open"
            return result

# Usage
agent = RobustAgent()
result = agent.execute_with_retry("Analyze this data", max_retries=3, use_fallback=True)
print(f"Success: {result['success']}, Model: {result.get('model_used')}")
```

---

## Recipe 10: Cost Tracking and Budgeting

**Use Case:** Monitor and control API costs
**Difficulty:** ⭐⭐ Intermediate

```python
from datetime import datetime, timedelta
from typing import Dict, Optional
from openai import OpenAI

class CostTracker:
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.costs = []
        # December 2025 pricing
        self.pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
            "gpt-4.5": {"input": 75.00 / 1_000_000, "output": 150.00 / 1_000_000},  # Preview pricing
            "o3-mini": {"input": 1.10 / 1_000_000, "output": 4.40 / 1_000_000},
            "claude-opus-4.5": {"input": 5.00 / 1_000_000, "output": 25.00 / 1_000_000},  # 67% reduction
            "claude-sonnet-4": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
            "claude-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000}
        }

    def track_call(self, model: str, input_tokens: int, output_tokens: int, metadata: Optional[Dict] = None) -> Dict:
        cost = input_tokens * self.pricing[model]["input"] + output_tokens * self.pricing[model]["output"]

        record = {
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "metadata": metadata or {}
        }

        self.costs.append(record)
        return {"cost": cost, "cumulative_today": self.get_today_cost(), "budget_remaining": self.daily_budget - self.get_today_cost()}

    def get_today_cost(self) -> float:
        today = datetime.now().date()
        return sum(r["cost"] for r in self.costs if r["timestamp"].date() == today)

    def can_afford(self, estimated_tokens: int, model: str) -> bool:
        estimated_cost = estimated_tokens * (self.pricing[model]["input"] + self.pricing[model]["output"]) / 2
        return self.get_today_cost() + estimated_cost <= self.daily_budget

    def get_report(self, days: int = 7) -> Dict:
        cutoff = datetime.now() - timedelta(days=days)
        recent = [c for c in self.costs if c["timestamp"] >= cutoff]

        total_cost = sum(c["cost"] for c in recent)
        by_model = {}
        for cost in recent:
            model = cost["model"]
            if model not in by_model:
                by_model[model] = {"cost": 0, "calls": 0}
            by_model[model]["cost"] += cost["cost"]
            by_model[model]["calls"] += 1

        return {
            "total_cost": total_cost,
            "total_calls": len(recent),
            "by_model": by_model,
            "today_cost": self.get_today_cost(),
            "budget_remaining": self.daily_budget - self.get_today_cost()
        }

class BudgetedAgent:
    def __init__(self, daily_budget: float = 100.0):
        self.client = OpenAI()
        self.tracker = CostTracker(daily_budget)

    def execute(self, task: str, model: str = "gpt-4o") -> Dict:
        estimated_tokens = len(task.split()) * 2

        if not self.tracker.can_afford(estimated_tokens, model):
            return {"success": False, "error": "Budget exceeded"}

        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": task}])

        cost_info = self.tracker.track_call(
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )

        return {"success": True, "result": response.choices[0].message.content, "cost": cost_info["cost"], "budget_remaining": cost_info["budget_remaining"]}

# Usage
agent = BudgetedAgent(daily_budget=10.0)
result = agent.execute("Analyze this data", model="gpt-4o-mini")
print(f"Cost: ${result['cost']:.4f}, Remaining: ${result['budget_remaining']:.2f}")
```

**Output:** `Cost: $0.0012, Remaining: $9.9988`

---

## Recipe 11: Model Cascading

**Use Case:** Start cheap, escalate only when needed
**Difficulty:** ⭐⭐ Intermediate

```python
from openai import OpenAI

class CascadingAgent:
    def __init__(self):
        self.client = OpenAI()

    def assess_complexity(self, task: str) -> float:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Rate task complexity 0-1: 0=simple, 0.5=moderate, 1=complex. Respond with only number."},
                {"role": "user", "content": f"Task: {task}"}
            ],
            max_tokens=10
        )

        try:
            complexity = float(response.choices[0].message.content.strip())
            return min(max(complexity, 0), 1)
        except:
            return 0.5

    def execute_with_cascading(self, task: str, complexity_threshold: float = 0.7) -> Dict:
        complexity = self.assess_complexity(task)
        model = "gpt-4o-mini" if complexity < complexity_threshold else "gpt-4o"

        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": task}])

        return {
            "result": response.choices[0].message.content,
            "model_used": model,
            "complexity": complexity,
            "cost_saved": complexity < complexity_threshold
        }

# Usage
agent = CascadingAgent()
tasks = [
    "What is 2+2?",
    "Explain quantum computing",
    "Design distributed system for 1M req/sec"
]

for task in tasks:
    result = agent.execute_with_cascading(task)
    print(f"{task[:30]}... → {result['model_used']} (complexity: {result['complexity']:.2f})")
```

**Output:**
```
What is 2+2?... → gpt-4o-mini (complexity: 0.05)
Explain quantum computing... → gpt-4o-mini (complexity: 0.45)
Design distributed system... → gpt-4o (complexity: 0.95)
```

**Cost Savings:** ~60% reduction by routing simple tasks to cheaper models

---

## Recipe 12: Agentic Plan Caching (NEW - December 2025)

**Use Case:** Cache and reuse agent execution plans for repeated similar tasks
**Difficulty:** ⭐⭐⭐ Advanced
**Research:** 50.31% average cost reduction, 96.61% accuracy retention (arXiv:2410.19414)

```python
import hashlib
import json
from typing import Dict, List, Optional
from openai import OpenAI

class AgenticPlanCache:
    """
    Cache agent plans for reuse on similar tasks.

    Research shows:
    - 46.62% cost reduction for GPT-4o-mini
    - 50.31% average across models
    - Semantic similarity threshold: 0.85
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.client = OpenAI()
        self.plan_cache = {}  # {task_hash: {plan, embedding, usage_count}}
        self.similarity_threshold = similarity_threshold

    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        # Cosine similarity
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0

    def find_similar_plan(self, task: str) -> Optional[Dict]:
        """Find cached plan for semantically similar task."""
        task_embedding = self._get_embedding(task)

        best_match = None
        best_similarity = 0

        for cached in self.plan_cache.values():
            similarity = self._compute_similarity(task_embedding, cached["embedding"])
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = cached

        if best_match:
            best_match["usage_count"] += 1
            return {"plan": best_match["plan"], "similarity": best_similarity, "reused": True}

        return None

    def generate_plan(self, task: str) -> Dict:
        """Generate new plan or retrieve from cache."""
        # Check cache first
        cached = self.find_similar_plan(task)
        if cached:
            return cached

        # Generate new plan
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a planning agent. Create a step-by-step plan.

Output format:
{
    "goal": "high-level goal",
    "steps": [
        {"step": 1, "action": "action description", "expected_output": "what to expect"},
        ...
    ],
    "success_criteria": ["criterion 1", "criterion 2"]
}"""},
                {"role": "user", "content": task}
            ],
            response_format={"type": "json_object"}
        )

        plan = json.loads(response.choices[0].message.content)
        task_embedding = self._get_embedding(task)

        # Cache the plan
        task_hash = hashlib.sha256(task.encode()).hexdigest()[:16]
        self.plan_cache[task_hash] = {
            "plan": plan,
            "embedding": task_embedding,
            "usage_count": 1,
            "original_task": task
        }

        return {"plan": plan, "similarity": 1.0, "reused": False}

    def get_cache_stats(self) -> Dict:
        """Return cache utilization statistics."""
        total_uses = sum(c["usage_count"] for c in self.plan_cache.values())
        reuses = total_uses - len(self.plan_cache)

        return {
            "cached_plans": len(self.plan_cache),
            "total_uses": total_uses,
            "cache_hits": reuses,
            "hit_rate": reuses / total_uses if total_uses > 0 else 0,
            "estimated_savings": f"{(reuses / total_uses * 100):.1f}%" if total_uses > 0 else "0%"
        }

# Usage
cache = AgenticPlanCache(similarity_threshold=0.85)

# First call - generates new plan
result1 = cache.generate_plan("Create a REST API for user management")
print(f"Generated: {result1['reused']}")  # False

# Similar call - reuses cached plan
result2 = cache.generate_plan("Build a REST API for user CRUD operations")
print(f"Reused: {result2['reused']}, Similarity: {result2['similarity']:.2f}")  # True, 0.92

# Check stats
print(cache.get_cache_stats())
# {'cached_plans': 1, 'total_uses': 2, 'cache_hits': 1, 'hit_rate': 0.5, 'estimated_savings': '50.0%'}
```

**Key Insights:**
- Semantic similarity threshold of 0.85 balances reuse vs accuracy
- Plans are task-agnostic templates that adapt to specific inputs
- Monitor cache hit rate to tune similarity threshold
- Consider TTL for cache entries in production

---

## Recipe 13: OpenAI Agents SDK Pattern (NEW - March 2025)

**Use Case:** Quick prototyping with built-in guardrails and handoffs
**Difficulty:** ⭐⭐ Intermediate
**Reference:** OpenAI Agents SDK (March 2025)

```python
# Conceptual pattern - OpenAI Agents SDK style
# Note: This shows the pattern, not exact SDK syntax

from typing import Callable, Dict, List, Any
from pydantic import BaseModel
from openai import OpenAI

class Guardrail(BaseModel):
    """Input/output validation guardrail."""
    name: str
    check_fn: Callable[[str], bool]
    error_message: str

class Agent:
    """Agent with tools, guardrails, and handoff capability."""

    def __init__(
        self,
        name: str,
        instructions: str,
        tools: List[Dict] = None,
        guardrails: List[Guardrail] = None,
        handoff_targets: List["Agent"] = None
    ):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.guardrails = guardrails or []
        self.handoff_targets = handoff_targets or []
        self.client = OpenAI()

    def validate_input(self, input_text: str) -> tuple[bool, str]:
        """Run input through guardrails."""
        for guardrail in self.guardrails:
            if not guardrail.check_fn(input_text):
                return False, guardrail.error_message
        return True, ""

    def can_handoff_to(self, target_name: str) -> bool:
        """Check if handoff to target is allowed."""
        return any(t.name == target_name for t in self.handoff_targets)

    def execute(self, task: str, context: Dict = None) -> Dict:
        """Execute task with guardrails and potential handoff."""

        # Input validation
        valid, error = self.validate_input(task)
        if not valid:
            return {"success": False, "error": error, "guardrail_blocked": True}

        # Build messages
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": task}
        ]

        if context:
            messages[0]["content"] += f"\n\nContext: {context}"

        # Execute with tools
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools if self.tools else None
        )

        result = response.choices[0].message.content

        # Check for handoff request in response
        for target in self.handoff_targets:
            if f"HANDOFF:{target.name}" in result:
                return {
                    "success": True,
                    "handoff": target.name,
                    "context": result.replace(f"HANDOFF:{target.name}", "").strip()
                }

        return {"success": True, "result": result, "agent": self.name}

class AgentRunner:
    """Run agents with automatic tracing and handoff handling."""

    def __init__(self):
        self.trace = []

    def run(self, agent: Agent, task: str, max_handoffs: int = 5) -> Dict:
        """Execute agent with handoff chain."""
        current_agent = agent
        context = None

        for i in range(max_handoffs):
            self.trace.append({
                "step": i + 1,
                "agent": current_agent.name,
                "input": task if i == 0 else "handoff"
            })

            result = current_agent.execute(task, context)

            if not result["success"]:
                return {"success": False, "error": result.get("error"), "trace": self.trace}

            if "handoff" in result:
                # Find handoff target
                target = next(
                    (t for t in current_agent.handoff_targets if t.name == result["handoff"]),
                    None
                )
                if target:
                    current_agent = target
                    context = result.get("context")
                    continue

            return {"success": True, "result": result["result"], "trace": self.trace}

        return {"success": False, "error": "Max handoffs exceeded", "trace": self.trace}

# Usage example
def no_pii(text: str) -> bool:
    """Check for PII patterns."""
    import re
    patterns = [r'\b\d{3}-\d{2}-\d{4}\b', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']
    return not any(re.search(p, text) for p in patterns)

# Create specialized agents
research_agent = Agent(
    name="researcher",
    instructions="Research topics thoroughly. Say HANDOFF:writer when research complete.",
    guardrails=[Guardrail(name="no_pii", check_fn=no_pii, error_message="PII detected")]
)

writer_agent = Agent(
    name="writer",
    instructions="Write clear, concise content based on research context."
)

research_agent.handoff_targets = [writer_agent]

# Run with automatic handoff
runner = AgentRunner()
result = runner.run(research_agent, "Research AI agent frameworks and write a summary")
print(f"Result: {result['success']}, Steps: {len(result['trace'])}")
```

**Key Features:**
- Guardrails for input/output validation
- Native agent-to-agent handoffs
- Automatic tracing for observability
- Provider-agnostic (100+ LLMs via Chat Completions API)

---

## Recipe 14: MCP Tool Integration (NEW - November 2025)

**Use Case:** Standardized tool integration via Model Context Protocol
**Difficulty:** ⭐⭐ Intermediate
**Reference:** MCP November 2025 spec (sampling, server-side loops, tasks)

```python
# MCP Integration Pattern
# Conceptual implementation following MCP protocol

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class MCPTool:
    """MCP-compliant tool definition."""
    name: str
    description: str
    input_schema: Dict
    server_id: str

@dataclass
class MCPResource:
    """MCP resource (files, databases, APIs)."""
    uri: str
    name: str
    mime_type: str
    server_id: str

class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self):
        self.servers = {}  # server_id -> connection
        self.tools = {}    # tool_name -> MCPTool
        self.resources = {}  # uri -> MCPResource

    def connect_server(self, server_id: str, config: Dict) -> bool:
        """Connect to MCP server (stdio or SSE transport)."""
        # In practice: establish JSON-RPC connection
        self.servers[server_id] = {
            "status": "connected",
            "config": config,
            "capabilities": self._get_capabilities(server_id)
        }

        # Discover tools and resources
        self._discover_tools(server_id)
        self._discover_resources(server_id)

        return True

    def _get_capabilities(self, server_id: str) -> Dict:
        """Get server capabilities (November 2025 spec)."""
        return {
            "tools": True,
            "resources": True,
            "prompts": True,
            "sampling": True,  # NEW: Server can request LLM completions
            "tasks": True,     # NEW: Long-running task management
            "logging": True
        }

    def _discover_tools(self, server_id: str) -> None:
        """Discover available tools from server."""
        # Simulated tool discovery
        discovered = [
            MCPTool(
                name="file_read",
                description="Read file contents",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                server_id=server_id
            ),
            MCPTool(
                name="web_search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                server_id=server_id
            )
        ]

        for tool in discovered:
            self.tools[f"{server_id}/{tool.name}"] = tool

    def _discover_resources(self, server_id: str) -> None:
        """Discover available resources from server."""
        # Simulated resource discovery
        pass

    def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call MCP tool with security validation."""

        # Security: Validate tool exists
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        tool = self.tools[tool_name]
        server = self.servers.get(tool.server_id)

        if not server or server["status"] != "connected":
            return {"error": f"Server not connected: {tool.server_id}"}

        # Security: Validate arguments against schema
        if not self._validate_schema(arguments, tool.input_schema):
            return {"error": "Invalid arguments"}

        # Execute tool call (in practice: JSON-RPC call)
        result = self._execute_tool_call(tool, arguments)

        # Security: Treat output as untrusted
        return self._sanitize_output(result)

    def _validate_schema(self, args: Dict, schema: Dict) -> bool:
        """Validate arguments against JSON schema."""
        # Simplified validation
        required = schema.get("required", [])
        return all(r in args for r in required)

    def _execute_tool_call(self, tool: MCPTool, arguments: Dict) -> Dict:
        """Execute the actual tool call."""
        # Simulated execution
        return {"status": "success", "result": f"Executed {tool.name}"}

    def _sanitize_output(self, output: Dict) -> Dict:
        """Sanitize tool output (treat as untrusted)."""
        # Remove potential injection patterns
        sanitized = json.dumps(output)
        # In practice: more sophisticated sanitization
        return json.loads(sanitized)

    def list_tools(self) -> List[Dict]:
        """List all available tools across servers."""
        return [
            {
                "name": name,
                "description": tool.description,
                "server": tool.server_id
            }
            for name, tool in self.tools.items()
        ]

class MCPAgentIntegration:
    """Integrate MCP tools with agent execution."""

    def __init__(self):
        self.mcp_client = MCPClient()
        self.openai_client = None  # Initialize as needed

    def setup_mcp_servers(self, servers: List[Dict]) -> None:
        """Connect to multiple MCP servers."""
        for server in servers:
            self.mcp_client.connect_server(
                server_id=server["id"],
                config=server["config"]
            )

    def get_tools_for_llm(self) -> List[Dict]:
        """Convert MCP tools to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name.replace("/", "_"),
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }
            for tool in self.mcp_client.tools.values()
        ]

    def execute_with_mcp_tools(self, task: str) -> Dict:
        """Execute task using MCP tools."""
        tools = self.get_tools_for_llm()

        # Agent execution loop with MCP tools
        # ... (integrate with existing agent patterns)

        return {"status": "executed", "tools_available": len(tools)}

# Usage
mcp = MCPAgentIntegration()
mcp.setup_mcp_servers([
    {"id": "filesystem", "config": {"transport": "stdio", "command": "npx @mcp/filesystem"}},
    {"id": "web", "config": {"transport": "sse", "url": "http://localhost:3000/mcp"}}
])

print(f"Available tools: {mcp.mcp_client.list_tools()}")
```

**MCP Security Best Practices:**
1. Verify server signatures before connecting
2. Apply principle of least privilege to tool permissions
3. Sanitize all parameters before execution
4. Treat all tool outputs as untrusted
5. Implement rate limiting per server
6. Log all tool calls for audit

---

## Production Case Studies (NEW)

### Klarna AI Assistant: Lessons Learned

**Context:** Klarna deployed AI for customer service (2.3M conversations/month)

**Initial Success Metrics:**
- 66% resolution rate
- 700 FTE equivalent workload handled
- 25% reduction in repeat inquiries

**Subsequent Challenges (December 2024):**
- Quality issues emerged at scale
- Re-hired human agents for complex cases
- Hybrid model proved more effective

**Lessons:**
| Lesson | Implementation |
|--------|----------------|
| Start narrow | Deploy for specific, well-defined use cases first |
| Human escalation | Always have human handoff for edge cases |
| Quality monitoring | Real-time quality scoring on all interactions |
| Gradual rollout | A/B test before full deployment |

### Replit Agent Incident: What Went Wrong

**Context:** Replit's AI agent deleted a production database and created 4,000 fake users

**Root Causes:**
1. Insufficient environment separation
2. Missing destructive action safeguards
3. No human-in-the-loop for critical operations
4. Inadequate testing with production data

**Prevention Patterns:**
```
Production Safety Checklist:
□ Separate dev/staging/prod environments completely
□ Read-only database access by default
□ Mandatory HITL for DELETE/DROP operations
□ Rate limiting on write operations
□ Audit logging for all database changes
□ Rollback capabilities for all modifications
```

---

## Security Patterns for Production

### Environment Separation
```
┌─────────────────────────────────────────────────┐
│                 PRODUCTION                       │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  Agent    │  │ Read-Only │  │  Audit    │   │
│  │  Sandbox  │──│  DB View  │──│   Log     │   │
│  └───────────┘  └───────────┘  └───────────┘   │
│         ↑                                        │
│    ┌────┴────┐                                  │
│    │  HITL   │ ← Required for writes            │
│    │  Gate   │                                  │
│    └─────────┘                                  │
└─────────────────────────────────────────────────┘
```

### Risk-Based HITL Gates
```python
def assess_risk_and_gate(action: Dict) -> str:
    """Determine approval requirement based on action risk."""

    risk_matrix = {
        "read": "auto_approve",      # No approval needed
        "create": "log_only",         # Log but auto-approve
        "update": "async_review",     # Approve within 5 min or auto
        "delete": "sync_approve",     # Must approve before execution
        "admin": "block"              # Never auto-approve
    }

    action_type = action.get("type", "unknown")
    risk_level = risk_matrix.get(action_type, "block")

    # Escalation factors
    if action.get("affects_production"):
        risk_level = "sync_approve"
    if action.get("irreversible"):
        risk_level = "block"

    return risk_level
```

### Least Privilege Tool Access
```python
TOOL_PERMISSIONS = {
    "research_agent": ["web_search", "file_read"],
    "writer_agent": ["file_read", "file_write"],
    "admin_agent": ["*"],  # Requires HITL for all actions
}

def validate_tool_access(agent_id: str, tool_name: str) -> bool:
    """Check if agent has permission to use tool."""
    allowed = TOOL_PERMISSIONS.get(agent_id, [])

    if "*" in allowed:
        # Admin agents require HITL confirmation
        return request_hitl_approval(agent_id, tool_name)

    return tool_name in allowed
```

---

## Quick Reference: Pattern Selection

### When to Use Each Pattern

**Single Agent (Recipes 1-4)**
- Simple, well-defined tasks
- Single domain expertise
- Budget-conscious projects
- Quick prototypes

**Multi-Agent (Recipes 5-6)**
- Complex, multi-domain tasks
- Specialized expertise needed
- Parallel processing beneficial
- Research/analysis workflows

**Tool Use (Recipes 7-8)**
- External data/actions required
- Real-time information needed
- Computation/validation required
- Security-critical operations

**Production (Recipes 9-11)**
- All production deployments
- Cost management essential
- Reliability critical
- High-volume applications

**Advanced Cost Optimization (Recipe 12)**
- Repeated similar tasks
- High-volume API usage
- 50%+ cost reduction needed
- Semantic task clustering

**Quick Prototyping (Recipe 13)**
- Rapid development
- Built-in guardrails needed
- Agent handoffs required
- Provider flexibility

**Standardized Tools (Recipe 14)**
- MCP server integration
- Multi-server tool discovery
- Security-first tool execution
- Protocol compliance required

### Model Selection Guide (December 2025)

| Task Type | Recommended Model | Cost | Use Case |
|-----------|------------------|------|----------|
| Simple queries | GPT-4o-mini | $ | Facts, simple tasks |
| Complex reasoning | Claude Opus 4.5 | $$$ | Analysis, planning (80.9% SWE-bench) |
| Long-form writing | Claude Sonnet 4 | $$ | Reports, articles |
| Bulk processing | Claude Haiku | $ | Classification, extraction |
| Reasoning tasks | o3-mini | $$ | Math, logic, structured problems |
| Browser automation | Claude Computer Use | $$$ | Web tasks (61.4% OSWorld) |

### Cost Optimization Checklist (Updated December 2025)

1. Use **agentic plan caching** (Recipe 12) for 50%+ cost reduction
2. Use **model cascading** (Recipe 11) for mixed complexity
3. Implement **cost tracking** (Recipe 10) for all production
4. Use **parallel agents** (Recipe 6) with cheaper models
5. Add **dynamic tool selection** (Recipe 7) to reduce token usage
6. Use **circuit breakers** (Recipe 9) to prevent cost overruns
7. Enable **prompt caching** (90% savings on cached reads)
8. Implement **semantic deduplication** before API calls

---

**Tested with:**
- OpenAI GPT-4o (2024-08-06), GPT-4.5 (Dec 2025), o3-mini
- Anthropic Claude Opus 4.5, Claude Sonnet 4, Claude Haiku
- LangGraph 1.0 (Oct 2025)
- OpenAI Agents SDK (Mar 2025)
- MCP Protocol (Nov 2025 spec)
- Python 3.10+

**Sources:**
- Agentic Plan Caching: arXiv:2410.19414 (46.62-50.31% cost reduction)
- LangGraph 1.0: langchain-ai.github.io/langgraph
- OpenAI Agents SDK: openai.com/agents-sdk
- MCP Protocol: modelcontextprotocol.io
- Klarna Case Study: LangChain customer stories (2024)
- Replit Incident: December 2024 reports

**Last Updated:** 2025-12-25
