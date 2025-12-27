# Tool Development Guide

**Last Updated:** December 2025
**Difficulty:** Intermediate to Advanced

Tools are the primary mechanism for agents to interact with external systems. This guide covers designing, implementing, and deploying production-grade tools for AI agents.

---

## Table of Contents

1. [Tool Design Principles](#tool-design-principles)
2. [Implementation Patterns](#implementation-patterns)
3. [Framework-Specific Formats](#framework-specific-formats)
4. [MCP Server Development](#mcp-server-development)
5. [Production Considerations](#production-considerations)
6. [Testing Tools](#testing-tools)
7. [Common Tool Patterns](#common-tool-patterns)

---

## Tool Design Principles

### Single Responsibility

Each tool should do one thing well:

```python
# Bad: Tool does too many things
@tool
def manage_user(action: str, user_id: str, data: dict) -> str:
    """Create, update, delete, or fetch user."""
    if action == "create":
        return create_user(data)
    elif action == "update":
        return update_user(user_id, data)
    # ... many more branches

# Good: Separate tools for each action
@tool
def create_user(name: str, email: str) -> str:
    """Create a new user with name and email."""
    return user_service.create(name=name, email=email)

@tool
def get_user(user_id: str) -> str:
    """Fetch user details by ID."""
    return user_service.get(user_id)

@tool
def update_user_email(user_id: str, new_email: str) -> str:
    """Update user's email address."""
    return user_service.update_email(user_id, new_email)
```

### Clear Input Schemas

Use Pydantic for explicit, validated schemas:

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
from datetime import date

class FlightSearchInput(BaseModel):
    """Input schema for flight search tool."""

    origin: str = Field(
        ...,
        description="Origin airport code (e.g., 'SFO', 'JFK')",
        min_length=3,
        max_length=4
    )
    destination: str = Field(
        ...,
        description="Destination airport code",
        min_length=3,
        max_length=4
    )
    departure_date: date = Field(
        ...,
        description="Departure date in YYYY-MM-DD format"
    )
    return_date: Optional[date] = Field(
        None,
        description="Return date for round trips"
    )
    cabin_class: Literal["economy", "business", "first"] = Field(
        "economy",
        description="Cabin class preference"
    )
    passengers: int = Field(
        1,
        ge=1,
        le=9,
        description="Number of passengers (1-9)"
    )

    @validator("destination")
    def destination_different_from_origin(cls, v, values):
        if "origin" in values and v == values["origin"]:
            raise ValueError("Destination must differ from origin")
        return v

    @validator("return_date")
    def return_after_departure(cls, v, values):
        if v and "departure_date" in values:
            if v < values["departure_date"]:
                raise ValueError("Return date must be after departure")
        return v
```

### Descriptive Outputs

Return structured, useful responses:

```python
from dataclasses import dataclass
from typing import List
import json

@dataclass
class FlightOption:
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    duration_minutes: int
    price_usd: float
    stops: int

@dataclass
class FlightSearchResult:
    query_summary: str
    options_found: int
    options: List[FlightOption]
    search_id: str  # For booking reference

    def to_tool_response(self) -> str:
        """Format for LLM consumption."""
        if self.options_found == 0:
            return f"No flights found for {self.query_summary}"

        response = f"Found {self.options_found} flights for {self.query_summary}:\n\n"
        for i, opt in enumerate(self.options[:5], 1):
            stops_text = "nonstop" if opt.stops == 0 else f"{opt.stops} stop(s)"
            response += (
                f"{i}. {opt.airline} {opt.flight_number}\n"
                f"   {opt.departure_time} - {opt.arrival_time} ({stops_text})\n"
                f"   ${opt.price_usd:.2f}\n\n"
            )

        if self.options_found > 5:
            response += f"... and {self.options_found - 5} more options"

        return response
```

### Comprehensive Error Handling

Tools must handle failures gracefully:

```python
from enum import Enum
from typing import Union

class ToolErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

@dataclass
class ToolError:
    error_type: ToolErrorType
    message: str
    retry_after_seconds: Optional[int] = None
    details: Optional[dict] = None

    def to_tool_response(self) -> str:
        response = f"Error: {self.message}"
        if self.retry_after_seconds:
            response += f" (retry after {self.retry_after_seconds}s)"
        return response

def search_flights(input: FlightSearchInput) -> Union[FlightSearchResult, ToolError]:
    """Search for available flights."""
    try:
        # Validate external dependencies
        if not flight_api.is_available():
            return ToolError(
                error_type=ToolErrorType.SERVICE_UNAVAILABLE,
                message="Flight search service temporarily unavailable",
                retry_after_seconds=60
            )

        # Make API call with timeout
        results = flight_api.search(
            origin=input.origin,
            destination=input.destination,
            date=input.departure_date,
            timeout=10
        )

        return FlightSearchResult(
            query_summary=f"{input.origin} to {input.destination} on {input.departure_date}",
            options_found=len(results),
            options=results,
            search_id=generate_search_id()
        )

    except flight_api.RateLimitError as e:
        return ToolError(
            error_type=ToolErrorType.RATE_LIMITED,
            message="Too many search requests",
            retry_after_seconds=e.retry_after
        )
    except flight_api.TimeoutError:
        return ToolError(
            error_type=ToolErrorType.TIMEOUT,
            message="Flight search timed out, please try again"
        )
    except Exception as e:
        logger.exception("Unexpected error in flight search")
        return ToolError(
            error_type=ToolErrorType.UNKNOWN,
            message="An unexpected error occurred"
        )
```

---

## Implementation Patterns

### OpenAI Function Calling Format

```python
import openai
import json
from typing import Callable, Dict, Any

# Tool definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location. Use this when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g., 'San Francisco, CA' or 'London, UK'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search for restaurants by cuisine type and location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or neighborhood"
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Type of cuisine (e.g., 'italian', 'japanese')"
                    },
                    "price_range": {
                        "type": "string",
                        "enum": ["$", "$$", "$$$", "$$$$"],
                        "description": "Price range"
                    },
                    "open_now": {
                        "type": "boolean",
                        "description": "Only show currently open restaurants"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Tool implementations
def get_weather(location: str, units: str = "fahrenheit") -> str:
    # Implementation
    weather_data = weather_api.get(location)
    temp = weather_data.temp_f if units == "fahrenheit" else weather_data.temp_c
    return f"Weather in {location}: {temp}° {units[0].upper()}, {weather_data.conditions}"

def search_restaurants(location: str, cuisine: str = None,
                       price_range: str = None, open_now: bool = False) -> str:
    results = restaurant_api.search(
        location=location,
        cuisine=cuisine,
        price=price_range,
        open_now=open_now
    )
    return json.dumps([r.to_dict() for r in results[:5]])

# Tool registry
TOOL_REGISTRY: Dict[str, Callable] = {
    "get_weather": get_weather,
    "search_restaurants": search_restaurants,
}

# Execution loop
def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # No tool calls - return final response
        if not message.tool_calls:
            return message.content

        # Process tool calls
        messages.append(message)

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            # Execute tool
            if func_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[func_name](**func_args)
            else:
                result = f"Error: Unknown tool {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
```

### Anthropic Tool Format

```python
import anthropic
from typing import List, Dict, Any

client = anthropic.Anthropic()

# Tool definitions
tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol. Returns the latest trading price and daily change.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL')"
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include 5-day price history"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "execute_trade",
        "description": "Execute a stock trade. Requires explicit user confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Trade action"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Number of shares",
                    "minimum": 1
                },
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit"],
                    "description": "Order type"
                },
                "limit_price": {
                    "type": "number",
                    "description": "Limit price (required for limit orders)"
                }
            },
            "required": ["ticker", "action", "quantity", "order_type"]
        }
    }
]

def process_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool and return result."""
    if tool_name == "get_stock_price":
        return get_stock_price(**tool_input)
    elif tool_name == "execute_trade":
        return execute_trade(**tool_input)
    else:
        return f"Unknown tool: {tool_name}"

def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # Check if we're done
        if response.stop_reason == "end_turn":
            # Extract text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Process tool use
        if response.stop_reason == "tool_use":
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({
                "role": "user",
                "content": tool_results
            })
```

### LangChain Tool Format

```python
from langchain_core.tools import tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
import httpx

# Method 1: @tool decorator
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Use this for any arithmetic calculations.
    Supports +, -, *, /, **, parentheses.

    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2', '(10 * 5) / 2')

    Returns:
        The calculated result as a string
    """
    try:
        # Safe evaluation (in production, use a proper math parser)
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Method 2: Pydantic schema with StructuredTool
class WebSearchInput(BaseModel):
    """Input for web search tool."""
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, ge=1, le=10, description="Number of results")
    site_filter: Optional[str] = Field(default=None, description="Limit to specific domain")

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

def web_search_impl(query: str, num_results: int = 5, site_filter: Optional[str] = None) -> str:
    """Execute web search."""
    search_query = f"site:{site_filter} {query}" if site_filter else query

    # Use search API
    results = search_api.search(search_query, limit=num_results)

    formatted = []
    for r in results:
        formatted.append(f"**{r.title}**\n{r.url}\n{r.snippet}\n")

    return "\n".join(formatted) if formatted else "No results found"

web_search = StructuredTool.from_function(
    func=web_search_impl,
    name="web_search",
    description="Search the web for current information. Use for recent events, facts, or data.",
    args_schema=WebSearchInput,
    return_direct=False
)


# Method 3: Async tools
class AsyncDatabaseTool:
    """Async tool for database operations."""

    def __init__(self, connection_pool):
        self.pool = connection_pool

    @tool
    async def query_database(self, sql: str, params: Optional[List] = None) -> str:
        """Execute a read-only SQL query against the database.

        Args:
            sql: SELECT query to execute
            params: Query parameters for prepared statement

        Returns:
            Query results as formatted table
        """
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *(params or []))

        if not rows:
            return "No results"

        # Format as table
        headers = list(rows[0].keys())
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows[:20]:
            lines.append(" | ".join(str(v) for v in row.values()))

        if len(rows) > 20:
            lines.append(f"... and {len(rows) - 20} more rows")

        return "\n".join(lines)


# Create agent with tools
def create_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    tools = [calculate, web_search]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to tools."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
```

---

## MCP Server Development

Model Context Protocol (MCP) enables standardized tool distribution across Claude environments.

### Python MCP Server

```python
# server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field
import httpx
import asyncio
from typing import Any

# Create server instance
server = Server("github-tools")

# Tool schemas
class SearchReposInput(BaseModel):
    query: str = Field(description="Search query for repositories")
    language: str | None = Field(default=None, description="Filter by programming language")
    sort: str = Field(default="stars", description="Sort by: stars, forks, updated")
    limit: int = Field(default=10, ge=1, le=100, description="Number of results")

class GetRepoInput(BaseModel):
    owner: str = Field(description="Repository owner")
    repo: str = Field(description="Repository name")

class ListIssuesInput(BaseModel):
    owner: str = Field(description="Repository owner")
    repo: str = Field(description="Repository name")
    state: str = Field(default="open", description="Filter by state: open, closed, all")
    labels: str | None = Field(default=None, description="Comma-separated label names")


# Register tools
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_repos",
            description="Search GitHub repositories by query, language, and other criteria",
            inputSchema=SearchReposInput.model_json_schema()
        ),
        Tool(
            name="get_repo",
            description="Get detailed information about a specific GitHub repository",
            inputSchema=GetRepoInput.model_json_schema()
        ),
        Tool(
            name="list_issues",
            description="List issues for a GitHub repository",
            inputSchema=ListIssuesInput.model_json_schema()
        )
    ]


# Tool implementations
@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}" if github_token else None
    }
    headers = {k: v for k, v in headers.items() if v}

    async with httpx.AsyncClient(base_url="https://api.github.com") as client:
        try:
            if name == "search_repos":
                result = await search_repos(client, headers, **arguments)
            elif name == "get_repo":
                result = await get_repo(client, headers, **arguments)
            elif name == "list_issues":
                result = await list_issues(client, headers, **arguments)
            else:
                result = f"Unknown tool: {name}"
        except httpx.HTTPError as e:
            result = f"GitHub API error: {str(e)}"

    return [TextContent(type="text", text=result)]


async def search_repos(client: httpx.AsyncClient, headers: dict,
                       query: str, language: str = None,
                       sort: str = "stars", limit: int = 10) -> str:
    """Search GitHub repositories."""
    q = query
    if language:
        q += f" language:{language}"

    response = await client.get(
        "/search/repositories",
        headers=headers,
        params={"q": q, "sort": sort, "per_page": limit}
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for repo in data.get("items", [])[:limit]:
        results.append(
            f"**{repo['full_name']}** ({repo['stargazers_count']} stars)\n"
            f"{repo['description'] or 'No description'}\n"
            f"URL: {repo['html_url']}\n"
        )

    return "\n".join(results) if results else "No repositories found"


async def get_repo(client: httpx.AsyncClient, headers: dict,
                   owner: str, repo: str) -> str:
    """Get repository details."""
    response = await client.get(f"/repos/{owner}/{repo}", headers=headers)
    response.raise_for_status()
    data = response.json()

    return (
        f"# {data['full_name']}\n\n"
        f"**Description:** {data['description'] or 'None'}\n"
        f"**Stars:** {data['stargazers_count']}\n"
        f"**Forks:** {data['forks_count']}\n"
        f"**Language:** {data['language']}\n"
        f"**Open Issues:** {data['open_issues_count']}\n"
        f"**Created:** {data['created_at'][:10]}\n"
        f"**Last Updated:** {data['updated_at'][:10]}\n"
        f"**URL:** {data['html_url']}\n"
    )


async def list_issues(client: httpx.AsyncClient, headers: dict,
                      owner: str, repo: str, state: str = "open",
                      labels: str = None) -> str:
    """List repository issues."""
    params = {"state": state, "per_page": 20}
    if labels:
        params["labels"] = labels

    response = await client.get(
        f"/repos/{owner}/{repo}/issues",
        headers=headers,
        params=params
    )
    response.raise_for_status()
    issues = response.json()

    if not issues:
        return f"No {state} issues found"

    results = [f"# Issues for {owner}/{repo}\n"]
    for issue in issues:
        if "pull_request" in issue:
            continue  # Skip PRs
        labels_str = ", ".join(l["name"] for l in issue["labels"]) if issue["labels"] else "none"
        results.append(
            f"- **#{issue['number']}** {issue['title']}\n"
            f"  State: {issue['state']} | Labels: {labels_str}\n"
        )

    return "\n".join(results)


# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

### TypeScript MCP Server

```typescript
// src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import Anthropic from "@anthropic-ai/sdk";

const server = new Server(
  { name: "claude-tools", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// Define tools
const tools: Tool[] = [
  {
    name: "analyze_sentiment",
    description: "Analyze the sentiment of text using Claude",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Text to analyze",
        },
        detailed: {
          type: "boolean",
          description: "Return detailed analysis with confidence scores",
          default: false,
        },
      },
      required: ["text"],
    },
  },
  {
    name: "summarize_document",
    description: "Summarize a document or long text",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Document text to summarize",
        },
        max_length: {
          type: "number",
          description: "Maximum summary length in words",
          default: 100,
        },
        style: {
          type: "string",
          enum: ["bullet_points", "paragraph", "executive"],
          description: "Summary style",
          default: "paragraph",
        },
      },
      required: ["text"],
    },
  },
];

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools,
}));

// Tool execution handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let result: string;

    switch (name) {
      case "analyze_sentiment":
        result = await analyzeSentiment(
          args.text as string,
          args.detailed as boolean
        );
        break;

      case "summarize_document":
        result = await summarizeDocument(
          args.text as string,
          args.max_length as number,
          args.style as string
        );
        break;

      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    return {
      content: [{ type: "text", text: result }],
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      content: [{ type: "text", text: `Error: ${message}` }],
      isError: true,
    };
  }
});

// Tool implementations
const anthropic = new Anthropic();

async function analyzeSentiment(
  text: string,
  detailed: boolean = false
): Promise<string> {
  const prompt = detailed
    ? `Analyze the sentiment of the following text. Provide:
1. Overall sentiment (positive/negative/neutral/mixed)
2. Confidence score (0-100%)
3. Key emotional indicators
4. Tone description

Text: ${text}`
    : `What is the sentiment of this text (positive/negative/neutral)? Text: ${text}`;

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 500,
    messages: [{ role: "user", content: prompt }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

async function summarizeDocument(
  text: string,
  maxLength: number = 100,
  style: string = "paragraph"
): Promise<string> {
  const styleInstructions: Record<string, string> = {
    bullet_points: "Format as bullet points with key takeaways",
    paragraph: "Write as a cohesive paragraph",
    executive: "Write an executive summary with context, findings, and recommendations",
  };

  const prompt = `Summarize the following text in approximately ${maxLength} words.
${styleInstructions[style] || styleInstructions.paragraph}

Text to summarize:
${text}`;

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: maxLength * 2,
    messages: [{ role: "user", content: prompt }],
  });

  return response.content[0].type === "text" ? response.content[0].text : "";
}

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP Server running on stdio");
}

main().catch(console.error);
```

### MCP Configuration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "github": {
      "command": "python",
      "args": ["/path/to/github-tools/server.py"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    },
    "claude-tools": {
      "command": "node",
      "args": ["/path/to/claude-tools/dist/index.js"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    },
    "database": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    }
  }
}
```

---

## Production Considerations

### Rate Limiting

```python
import time
from functools import wraps
from collections import defaultdict
import threading

class RateLimiter:
    """Token bucket rate limiter for tools."""

    def __init__(self, requests_per_minute: int = 60):
        self.rate = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.rate,
                self.tokens + time_passed * (self.rate / 60)
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait(self) -> float:
        """Wait until a token is available. Returns wait time."""
        while not self.acquire():
            wait_time = (1 - self.tokens) * (60 / self.rate)
            time.sleep(min(wait_time, 1))
        return 0


# Per-user rate limiting
class UserRateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.limiters: dict[str, RateLimiter] = defaultdict(
            lambda: RateLimiter(requests_per_minute)
        )

    def check(self, user_id: str) -> bool:
        return self.limiters[user_id].acquire()


def rate_limited(limiter: RateLimiter):
    """Decorator for rate-limited tools."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire():
                return ToolError(
                    error_type=ToolErrorType.RATE_LIMITED,
                    message="Rate limit exceeded",
                    retry_after_seconds=2
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Usage
api_limiter = RateLimiter(requests_per_minute=100)

@rate_limited(api_limiter)
@tool
def external_api_call(query: str) -> str:
    """Make rate-limited external API call."""
    return api.search(query)
```

### Caching

```python
from functools import lru_cache
import hashlib
import json
import redis
from datetime import timedelta

class ToolCache:
    """Redis-based cache for tool results."""

    def __init__(self, redis_url: str, default_ttl: int = 300):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _make_key(self, tool_name: str, args: dict) -> str:
        args_hash = hashlib.sha256(
            json.dumps(args, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"tool:{tool_name}:{args_hash}"

    def get(self, tool_name: str, args: dict) -> str | None:
        key = self._make_key(tool_name, args)
        cached = self.redis.get(key)
        return cached.decode() if cached else None

    def set(self, tool_name: str, args: dict, result: str, ttl: int = None):
        key = self._make_key(tool_name, args)
        self.redis.setex(key, ttl or self.default_ttl, result)

    def invalidate(self, tool_name: str, args: dict = None):
        if args:
            self.redis.delete(self._make_key(tool_name, args))
        else:
            # Invalidate all cached results for this tool
            pattern = f"tool:{tool_name}:*"
            for key in self.redis.scan_iter(pattern):
                self.redis.delete(key)


def cached_tool(cache: ToolCache, ttl: int = 300):
    """Decorator for cached tools."""
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            tool_name = func.__name__

            # Check cache
            cached = cache.get(tool_name, kwargs)
            if cached:
                return cached

            # Execute and cache
            result = func(**kwargs)
            if not isinstance(result, ToolError):
                cache.set(tool_name, kwargs, result, ttl)

            return result
        return wrapper
    return decorator


# Usage
cache = ToolCache("redis://localhost:6379")

@cached_tool(cache, ttl=600)
@tool
def get_company_info(ticker: str) -> str:
    """Get company information (cached for 10 minutes)."""
    return company_api.get_info(ticker)
```

### Monitoring

```python
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import logging
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
TOOL_CALLS = Counter(
    "tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"]
)

TOOL_LATENCY = Histogram(
    "tool_latency_seconds",
    "Tool execution latency",
    ["tool_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

TOOL_ERRORS = Counter(
    "tool_errors_total",
    "Tool errors by type",
    ["tool_name", "error_type"]
)


@dataclass
class ToolMetrics:
    """Metrics for a single tool invocation."""
    tool_name: str
    start_time: datetime
    end_time: datetime = None
    duration_ms: float = 0
    success: bool = True
    error_type: str = None
    input_size: int = 0
    output_size: int = 0


class ToolMonitor:
    """Monitor tool invocations."""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.metrics: List[ToolMetrics] = []

    def record(self, metrics: ToolMetrics):
        """Record metrics for a tool invocation."""
        self.metrics.append(metrics)

        # Update Prometheus
        status = "success" if metrics.success else "error"
        TOOL_CALLS.labels(tool_name=metrics.tool_name, status=status).inc()
        TOOL_LATENCY.labels(tool_name=metrics.tool_name).observe(
            metrics.duration_ms / 1000
        )

        if not metrics.success:
            TOOL_ERRORS.labels(
                tool_name=metrics.tool_name,
                error_type=metrics.error_type or "unknown"
            ).inc()

        # Log
        self.logger.info(
            "Tool invocation",
            extra={
                "tool": metrics.tool_name,
                "duration_ms": metrics.duration_ms,
                "success": metrics.success,
                "error_type": metrics.error_type
            }
        )


def monitored(monitor: ToolMonitor):
    """Decorator for monitored tools."""
    def decorator(func):
        @wraps(func)
        def wrapper(**kwargs):
            metrics = ToolMetrics(
                tool_name=func.__name__,
                start_time=datetime.utcnow(),
                input_size=len(str(kwargs))
            )

            try:
                result = func(**kwargs)
                metrics.success = not isinstance(result, ToolError)
                if isinstance(result, ToolError):
                    metrics.error_type = result.error_type.value
                metrics.output_size = len(str(result))
                return result

            except Exception as e:
                metrics.success = False
                metrics.error_type = type(e).__name__
                raise

            finally:
                metrics.end_time = datetime.utcnow()
                metrics.duration_ms = (
                    metrics.end_time - metrics.start_time
                ).total_seconds() * 1000
                monitor.record(metrics)

        return wrapper
    return decorator
```

### Versioning

```python
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from functools import wraps

@dataclass
class ToolVersion:
    version: str
    implementation: Callable
    deprecated: bool = False
    deprecation_message: str = None


class VersionedToolRegistry:
    """Manage multiple versions of tools."""

    def __init__(self):
        self.tools: Dict[str, Dict[str, ToolVersion]] = {}
        self.default_versions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        version: str,
        implementation: Callable,
        is_default: bool = False,
        deprecated: bool = False,
        deprecation_message: str = None
    ):
        """Register a tool version."""
        if name not in self.tools:
            self.tools[name] = {}

        self.tools[name][version] = ToolVersion(
            version=version,
            implementation=implementation,
            deprecated=deprecated,
            deprecation_message=deprecation_message
        )

        if is_default:
            self.default_versions[name] = version

    def get(self, name: str, version: str = None) -> Optional[ToolVersion]:
        """Get a tool by name and optional version."""
        if name not in self.tools:
            return None

        version = version or self.default_versions.get(name)
        if not version:
            # Return latest version
            version = max(self.tools[name].keys())

        return self.tools[name].get(version)

    def invoke(self, name: str, version: str = None, **kwargs):
        """Invoke a tool with version resolution."""
        tool = self.get(name, version)
        if not tool:
            raise ValueError(f"Tool not found: {name} v{version}")

        if tool.deprecated:
            import warnings
            warnings.warn(
                tool.deprecation_message or f"{name} v{tool.version} is deprecated",
                DeprecationWarning
            )

        return tool.implementation(**kwargs)


# Usage example
registry = VersionedToolRegistry()

# V1 - Original implementation
def search_v1(query: str) -> str:
    return legacy_search(query)

# V2 - Improved implementation
def search_v2(query: str, filters: dict = None) -> str:
    return improved_search(query, filters)

registry.register("search", "1.0", search_v1, deprecated=True,
                  deprecation_message="Use search v2.0 for better results")
registry.register("search", "2.0", search_v2, is_default=True)

# Invoke latest
result = registry.invoke("search", query="AI agents")

# Invoke specific version
result = registry.invoke("search", version="1.0", query="AI agents")
```

---

## Testing Tools

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import date

class TestFlightSearchTool:
    """Unit tests for flight search tool."""

    @pytest.fixture
    def mock_flight_api(self):
        with patch("tools.flights.flight_api") as mock:
            mock.is_available.return_value = True
            yield mock

    @pytest.fixture
    def valid_input(self):
        return FlightSearchInput(
            origin="SFO",
            destination="JFK",
            departure_date=date(2025, 3, 15),
            passengers=2
        )

    def test_successful_search(self, mock_flight_api, valid_input):
        """Test successful flight search."""
        mock_flight_api.search.return_value = [
            FlightOption(
                airline="United",
                flight_number="UA123",
                departure_time="08:00",
                arrival_time="16:30",
                duration_minutes=330,
                price_usd=450.00,
                stops=0
            )
        ]

        result = search_flights(valid_input)

        assert isinstance(result, FlightSearchResult)
        assert result.options_found == 1
        assert "SFO to JFK" in result.query_summary

    def test_no_results(self, mock_flight_api, valid_input):
        """Test empty results handling."""
        mock_flight_api.search.return_value = []

        result = search_flights(valid_input)

        assert result.options_found == 0
        assert "No flights found" in result.to_tool_response()

    def test_service_unavailable(self, mock_flight_api, valid_input):
        """Test graceful handling when service is down."""
        mock_flight_api.is_available.return_value = False

        result = search_flights(valid_input)

        assert isinstance(result, ToolError)
        assert result.error_type == ToolErrorType.SERVICE_UNAVAILABLE
        assert result.retry_after_seconds is not None

    def test_rate_limit_handling(self, mock_flight_api, valid_input):
        """Test rate limit error handling."""
        mock_flight_api.search.side_effect = flight_api.RateLimitError(
            retry_after=60
        )

        result = search_flights(valid_input)

        assert isinstance(result, ToolError)
        assert result.error_type == ToolErrorType.RATE_LIMITED
        assert result.retry_after_seconds == 60

    def test_timeout_handling(self, mock_flight_api, valid_input):
        """Test timeout error handling."""
        mock_flight_api.search.side_effect = flight_api.TimeoutError()

        result = search_flights(valid_input)

        assert isinstance(result, ToolError)
        assert result.error_type == ToolErrorType.TIMEOUT


class TestInputValidation:
    """Test input validation."""

    def test_valid_input(self):
        """Test valid input passes validation."""
        input = FlightSearchInput(
            origin="SFO",
            destination="JFK",
            departure_date=date(2025, 3, 15)
        )
        assert input.origin == "SFO"

    def test_same_origin_destination_rejected(self):
        """Test same origin and destination is rejected."""
        with pytest.raises(ValueError, match="must differ"):
            FlightSearchInput(
                origin="SFO",
                destination="SFO",
                departure_date=date(2025, 3, 15)
            )

    def test_return_before_departure_rejected(self):
        """Test return date before departure is rejected."""
        with pytest.raises(ValueError, match="after departure"):
            FlightSearchInput(
                origin="SFO",
                destination="JFK",
                departure_date=date(2025, 3, 15),
                return_date=date(2025, 3, 10)
            )

    def test_invalid_airport_code(self):
        """Test invalid airport code is rejected."""
        with pytest.raises(ValueError):
            FlightSearchInput(
                origin="X",  # Too short
                destination="JFK",
                departure_date=date(2025, 3, 15)
            )
```

### Integration Testing

```python
import pytest
from httpx import AsyncClient
import respx

class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    @pytest.fixture
    async def mcp_client(self):
        """Create test client for MCP server."""
        from mcp.client import Client
        client = Client()
        await client.connect("stdio", ["python", "server.py"])
        yield client
        await client.disconnect()

    @pytest.mark.asyncio
    @respx.mock
    async def test_search_repos_integration(self, mcp_client):
        """Test GitHub search integration."""
        # Mock GitHub API
        respx.get("https://api.github.com/search/repositories").mock(
            return_value=httpx.Response(200, json={
                "items": [{
                    "full_name": "langchain-ai/langchain",
                    "description": "LangChain framework",
                    "stargazers_count": 50000,
                    "html_url": "https://github.com/langchain-ai/langchain"
                }]
            })
        )

        result = await mcp_client.call_tool(
            "search_repos",
            {"query": "langchain", "limit": 5}
        )

        assert "langchain-ai/langchain" in result
        assert "50000" in result

    @pytest.mark.asyncio
    async def test_tool_error_propagation(self, mcp_client):
        """Test that tool errors are properly propagated."""
        result = await mcp_client.call_tool(
            "get_repo",
            {"owner": "nonexistent", "repo": "nonexistent"}
        )

        assert "Error" in result or "not found" in result.lower()
```

---

## Common Tool Patterns

### Database Query Tool

```python
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import re

class DatabaseQueryTool:
    """Safe database query tool with SQL injection prevention."""

    DANGEROUS_PATTERNS = [
        r"\bDROP\b", r"\bDELETE\b", r"\bTRUNCATE\b",
        r"\bUPDATE\b", r"\bINSERT\b", r"\bALTER\b",
        r"\bCREATE\b", r"\bGRANT\b", r"\bREVOKE\b",
        r"--", r";.*;"
    ]

    def __init__(self, connection_string: str, max_rows: int = 100):
        self.engine = create_engine(connection_string)
        self.max_rows = max_rows

    def validate_query(self, sql: str) -> tuple[bool, str]:
        """Validate query is safe to execute."""
        sql_upper = sql.upper()

        # Must be SELECT
        if not sql_upper.strip().startswith("SELECT"):
            return False, "Only SELECT queries are allowed"

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper):
                return False, f"Query contains forbidden pattern"

        return True, ""

    def execute(self, sql: str, params: dict = None) -> str:
        """Execute a read-only query."""
        # Validate
        is_valid, error = self.validate_query(sql)
        if not is_valid:
            return f"Query rejected: {error}"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(sql + f" LIMIT {self.max_rows}"),
                    params or {}
                )
                rows = result.fetchall()
                columns = result.keys()

            if not rows:
                return "Query returned no results"

            # Format as markdown table
            header = " | ".join(columns)
            separator = " | ".join("---" for _ in columns)
            data_rows = [
                " | ".join(str(v) for v in row)
                for row in rows
            ]

            table = "\n".join([header, separator] + data_rows)

            if len(rows) == self.max_rows:
                table += f"\n\n*Results limited to {self.max_rows} rows*"

            return table

        except SQLAlchemyError as e:
            return f"Database error: {str(e)}"
```

### File Manipulation Tool

```python
import os
from pathlib import Path
from typing import Literal

class SecureFileTool:
    """Secure file operations within allowed directories."""

    def __init__(self, allowed_dirs: list[str], max_file_size: int = 10_000_000):
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.max_file_size = max_file_size

    def _validate_path(self, path: str) -> tuple[bool, Path, str]:
        """Validate path is within allowed directories."""
        try:
            resolved = Path(path).resolve()

            # Check if within allowed directories
            if not any(
                str(resolved).startswith(str(allowed))
                for allowed in self.allowed_dirs
            ):
                return False, resolved, "Path outside allowed directories"

            return True, resolved, ""

        except Exception as e:
            return False, None, str(e)

    def read_file(self, path: str) -> str:
        """Read file contents."""
        valid, resolved, error = self._validate_path(path)
        if not valid:
            return f"Error: {error}"

        if not resolved.exists():
            return f"Error: File not found: {path}"

        if not resolved.is_file():
            return f"Error: Not a file: {path}"

        if resolved.stat().st_size > self.max_file_size:
            return f"Error: File too large (max {self.max_file_size} bytes)"

        try:
            return resolved.read_text()
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to file."""
        valid, resolved, error = self._validate_path(path)
        if not valid:
            return f"Error: {error}"

        if len(content) > self.max_file_size:
            return f"Error: Content too large (max {self.max_file_size} bytes)"

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def list_directory(self, path: str) -> str:
        """List directory contents."""
        valid, resolved, error = self._validate_path(path)
        if not valid:
            return f"Error: {error}"

        if not resolved.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            entries = []
            for entry in sorted(resolved.iterdir()):
                if entry.is_dir():
                    entries.append(f"[DIR]  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    entries.append(f"[FILE] {entry.name} ({size} bytes)")

            return "\n".join(entries) if entries else "Directory is empty"
        except Exception as e:
            return f"Error listing directory: {e}"
```

### API Integration Tool

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

class APIClient:
    """Robust API client with retries and circuit breaker."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout
        )
        self.max_retries = max_retries
        self._failure_count = 0
        self._circuit_open = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> dict:
        """Make API request with retry logic."""
        if self._circuit_open:
            raise Exception("Circuit breaker open - service unavailable")

        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            self._failure_count = 0
            return response.json()

        except httpx.HTTPStatusError as e:
            self._failure_count += 1
            if self._failure_count >= 5:
                self._circuit_open = True
            raise

    async def get(self, path: str, params: dict = None) -> dict:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, data: dict = None) -> dict:
        return await self.request("POST", path, json=data)


# Weather API tool example
class WeatherTool:
    def __init__(self, api_key: str):
        self.client = APIClient(
            base_url="https://api.weatherapi.com/v1",
            api_key=api_key
        )

    async def get_current(self, location: str) -> str:
        """Get current weather for location."""
        try:
            data = await self.client.get(
                "/current.json",
                params={"q": location}
            )

            current = data["current"]
            location_info = data["location"]

            return (
                f"Weather in {location_info['name']}, {location_info['country']}:\n"
                f"Temperature: {current['temp_f']}°F ({current['temp_c']}°C)\n"
                f"Condition: {current['condition']['text']}\n"
                f"Humidity: {current['humidity']}%\n"
                f"Wind: {current['wind_mph']} mph {current['wind_dir']}"
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return f"Location not found: {location}"
            return f"Weather API error: {e.response.status_code}"
        except Exception as e:
            return f"Error fetching weather: {str(e)}"
```

---

## Related Documents

- [Agent Prompting Guide](agent-prompting-guide.md) - Prompt design for tool-using agents
- [Memory Systems Guide](memory-systems-guide.md) - State management for agents
- [MCP Deep Dive](../phase-6-advanced/mcp-deep-dive.md) - Advanced MCP development
- [Testing Guide](../phase-4-production/testing-guide.md) - Testing agent tools
- [Security Essentials](../phase-5-security-compliance/security-essentials.md) - Tool security
