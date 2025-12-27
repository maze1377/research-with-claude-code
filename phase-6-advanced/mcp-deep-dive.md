# MCP Deep Dive: Model Context Protocol for Agentic AI

**Comprehensive guide to building production-ready MCP servers and integrations**

**Last Updated:** 2025-12-27

---

## December 2025 Key Developments

| Milestone | Significance | Date |
|-----------|--------------|------|
| **AAIF Formation** | MCP donated to Linux Foundation's Agentic AI Foundation | 2025-12-09 |
| **Protocol v2025-11-25** | Current stable specification with OAuth 2.1 | 2025-11-25 |
| **Streamable HTTP** | SSE deprecated, replaced by streamable HTTP transport | 2025-06-18 |
| **10,000+ Servers** | Active public MCP servers in production | 2025-12 |
| **97M+ Downloads** | Monthly SDK downloads across Python and TypeScript | 2025-12 |

**Founding Members (AAIF):** Anthropic, Block, OpenAI
**Platinum Supporters:** AWS, Google, Microsoft, Cloudflare, Bloomberg

Sources:
- [Linux Foundation AAIF Announcement](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
- [Anthropic MCP Donation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- [MCP Joins AAIF Blog](http://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/)

---

## 1. MCP Architecture

### 1.1 Protocol Overview

The Model Context Protocol (MCP) is a standardized protocol for connecting AI applications to external tools, data sources, and services. Built on JSON-RPC 2.0, it provides a universal interface for AI agents to interact with the world.

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Host                                │
│                (Claude Desktop, VS Code)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │         │
│  │  (weather)  │  │ (database)  │  │ (filesystem)│         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
     ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
     │   MCP   │      │   MCP   │      │   MCP   │
     │ Server  │      │ Server  │      │ Server  │
     │(weather)│      │(postgres)│     │ (local) │
     └─────────┘      └─────────┘      └─────────┘
```

**Key Participants:**

| Component | Role | Example |
|-----------|------|---------|
| **MCP Host** | AI application that coordinates multiple clients | Claude Desktop, VS Code, Cursor |
| **MCP Client** | Maintains connection to a single server | Built into host application |
| **MCP Server** | Exposes capabilities via standardized protocol | Filesystem, database, API servers |

### 1.2 Two-Layer Architecture

**Layer 1: Data Layer (JSON-RPC 2.0)**

Defines the protocol semantics:
- Lifecycle management (initialization, capability negotiation, termination)
- Server features (tools, resources, prompts)
- Client features (sampling, elicitation, logging)
- Utility features (notifications, progress tracking)

**Layer 2: Transport Layer**

Manages communication channels:

| Transport | Use Case | Features |
|-----------|----------|----------|
| **stdio** | Local processes | Direct I/O streams, optimal performance, no network overhead |
| **Streamable HTTP** | Remote servers | HTTP POST for requests, optional SSE for streaming, OAuth/API keys |

### 1.3 Protocol Versioning

MCP uses date-based version identifiers: `YYYY-MM-DD`

```
Current: 2025-11-25
Previous: 2025-06-18, 2025-03-26, 2024-11-05
```

**Version Negotiation:**
- Occurs during initialization
- Clients and servers MAY support multiple versions
- Must agree on single version for session
- Backward-compatible changes do not increment version

### 1.4 Message Formats

**Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "San Francisco"
    }
  }
}
```

**Response Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Current weather: 72F, sunny"
      }
    ]
  }
}
```

**Error Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Unknown tool: invalid_tool_name"
  }
}
```

**Notification Format (no response expected):**
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/tools/list_changed"
}
```

### 1.5 Connection Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│  1. INITIALIZE: Capability negotiation                        │
│     Client: initialize request with capabilities             │
│     Server: initialize response with capabilities            │
│     Client: notifications/initialized                        │
│                                                              │
│  2. DISCOVERY: Learn available features                      │
│     tools/list    -> Available tools                         │
│     resources/list -> Available resources                    │
│     prompts/list  -> Available prompts                       │
│                                                              │
│  3. OPERATION: Execute tools, read resources                 │
│     tools/call    -> Execute tool                            │
│     resources/read -> Read resource                          │
│     prompts/get   -> Get prompt template                     │
│                                                              │
│  4. NOTIFICATIONS: Real-time updates                         │
│     notifications/tools/list_changed                         │
│     notifications/resources/updated                          │
│                                                              │
│  5. SHUTDOWN: Graceful termination                           │
│     Close transport connection                               │
└──────────────────────────────────────────────────────────────┘
```

**Initialization Exchange:**

```json
// Client Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "sampling": {}
    },
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.0.0"
    }
  }
}

// Server Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": true }
    },
    "serverInfo": {
      "name": "weather-server",
      "version": "1.0.0"
    }
  }
}

// Client Notification
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

### 1.6 Core Primitives

| Primitive | Control | Purpose | Discovery |
|-----------|---------|---------|-----------|
| **Tools** | Model-controlled | Executable functions LLM can invoke | `tools/list` |
| **Resources** | Application-controlled | Read-only data sources for context | `resources/list` |
| **Prompts** | User-controlled | Pre-built instruction templates | `prompts/list` |

---

## 2. Building MCP Servers

### 2.1 Python SDK (FastMCP)

**Installation:**
```bash
# Using uv (recommended)
uv add "mcp[cli]"

# Using pip
pip install "mcp[cli]"
```

**Complete Weather Server Example:**

```python
# weather_server.py
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get("event", "Unknown")}
Area: {props.get("areaDesc", "Unknown")}
Severity: {props.get("severity", "Unknown")}
Description: {props.get("description", "No description")}
Instructions: {props.get("instruction", "None provided")}
"""


@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]} {period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


@mcp.resource("weather://current/{city}")
async def get_current_weather(city: str) -> str:
    """Get current weather conditions for a city."""
    # Implementation would use a weather API
    return f"Current conditions for {city}: 72F, sunny"


@mcp.prompt()
def weather_report(location: str) -> str:
    """Generate a weather report prompt for a location."""
    return f"""Please provide a comprehensive weather report for {location}.
Include:
1. Current conditions
2. Today's forecast
3. Any active weather alerts
4. Recommendations for outdoor activities
"""


def main():
    # Run the server with stdio transport
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

**Run the server:**
```bash
uv run weather_server.py
```

### 2.2 TypeScript SDK

**Installation:**
```bash
npm install @modelcontextprotocol/server zod
```

**Complete Weather Server Example:**

```typescript
// weather_server.ts
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

const NWS_API_BASE = "https://api.weather.gov";
const USER_AGENT = "weather-app/1.0";

// Create server instance
const server = new McpServer({
  name: "weather",
  version: "1.0.0",
});

// Helper function for API requests
async function makeNWSRequest(url: string): Promise<any | null> {
  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
      },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

// Format alert data
function formatAlert(feature: any): string {
  const props = feature.properties;
  return `
Event: ${props.event || "Unknown"}
Area: ${props.areaDesc || "Unknown"}
Severity: ${props.severity || "Unknown"}
Description: ${props.description || "No description"}
Instructions: ${props.instruction || "None provided"}
`;
}

// Register tools
server.tool(
  "get_alerts",
  "Get weather alerts for a US state",
  {
    state: z.string().describe("Two-letter US state code (e.g. CA, NY)"),
  },
  async ({ state }) => {
    const url = `${NWS_API_BASE}/alerts/active/area/${state}`;
    const data = await makeNWSRequest(url);

    if (!data || !data.features) {
      return {
        content: [
          { type: "text", text: "Unable to fetch alerts or no alerts found." },
        ],
      };
    }

    if (data.features.length === 0) {
      return {
        content: [
          { type: "text", text: "No active alerts for this state." },
        ],
      };
    }

    const alerts = data.features.map(formatAlert).join("\n---\n");
    return {
      content: [{ type: "text", text: alerts }],
    };
  }
);

server.tool(
  "get_forecast",
  "Get weather forecast for a location",
  {
    latitude: z.number().describe("Latitude of the location"),
    longitude: z.number().describe("Longitude of the location"),
  },
  async ({ latitude, longitude }) => {
    // Get forecast grid endpoint
    const pointsUrl = `${NWS_API_BASE}/points/${latitude},${longitude}`;
    const pointsData = await makeNWSRequest(pointsUrl);

    if (!pointsData) {
      return {
        content: [
          { type: "text", text: "Unable to fetch forecast data for this location." },
        ],
      };
    }

    // Get detailed forecast
    const forecastUrl = pointsData.properties.forecast;
    const forecastData = await makeNWSRequest(forecastUrl);

    if (!forecastData) {
      return {
        content: [
          { type: "text", text: "Unable to fetch detailed forecast." },
        ],
      };
    }

    // Format periods
    const periods = forecastData.properties.periods.slice(0, 5);
    const forecasts = periods.map((period: any) => `
${period.name}:
Temperature: ${period.temperature}${period.temperatureUnit}
Wind: ${period.windSpeed} ${period.windDirection}
Forecast: ${period.detailedForecast}
`).join("\n---\n");

    return {
      content: [{ type: "text", text: forecasts }],
    };
  }
);

// Register resources
server.resource(
  "weather://current/{city}",
  "Get current weather conditions for a city",
  async (uri) => {
    const city = uri.pathname.split("/").pop();
    return {
      contents: [
        {
          uri: uri.href,
          mimeType: "text/plain",
          text: `Current conditions for ${city}: 72F, sunny`,
        },
      ],
    };
  }
);

// Register prompts
server.prompt(
  "weather_report",
  "Generate a weather report prompt",
  {
    location: z.string().describe("Location for the weather report"),
  },
  async ({ location }) => {
    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Please provide a comprehensive weather report for ${location}.
Include:
1. Current conditions
2. Today's forecast
3. Any active weather alerts
4. Recommendations for outdoor activities`,
          },
        },
      ],
    };
  }
);

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP server running on stdio");
}

main().catch(console.error);
```

**Run the server:**
```bash
npx tsx weather_server.ts
```

### 2.3 Server Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     Server Lifecycle                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. STARTUP                                                 │
│     - Load configuration                                    │
│     - Initialize resources (DB connections, API clients)    │
│     - Register tools, resources, prompts                    │
│     - Start transport listener                              │
│                                                             │
│  2. INITIALIZATION (per client)                             │
│     - Receive initialize request                            │
│     - Negotiate protocol version                            │
│     - Exchange capabilities                                 │
│     - Receive initialized notification                      │
│                                                             │
│  3. OPERATION                                               │
│     - Handle requests (tools/call, resources/read, etc.)    │
│     - Send notifications (list changes, updates)            │
│     - Manage subscriptions                                  │
│                                                             │
│  4. SHUTDOWN                                                │
│     - Close client connections                              │
│     - Release resources                                     │
│     - Exit gracefully                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Tool Registration Patterns

**Python with Decorators:**
```python
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel

mcp = FastMCP("my-server")

# Simple tool with type hints
@mcp.tool()
async def simple_tool(name: str, count: int = 10) -> str:
    """A simple tool with type hints.

    Args:
        name: The name to process
        count: Number of times to repeat (default: 10)
    """
    return f"Hello, {name}! " * count

# Tool with context for logging/progress
@mcp.tool()
async def tool_with_context(query: str, ctx: Context) -> str:
    """Tool that uses context for logging and progress."""
    ctx.info(f"Processing query: {query}")
    ctx.report_progress(0.5, "Halfway done...")
    result = await process_query(query)
    ctx.report_progress(1.0, "Complete")
    return result

# Tool with structured output
class SearchResult(BaseModel):
    title: str
    url: str
    score: float

@mcp.tool()
async def search(query: str) -> list[SearchResult]:
    """Search and return structured results."""
    return [
        SearchResult(title="Result 1", url="https://example.com", score=0.95)
    ]
```

**TypeScript with Zod Schemas:**
```typescript
import { z } from "zod";

// Input schema with validation
const SearchInputSchema = z.object({
  query: z.string().min(1).describe("Search query"),
  limit: z.number().min(1).max(100).default(10).describe("Max results"),
  filters: z.object({
    category: z.string().optional(),
    dateFrom: z.string().datetime().optional(),
  }).optional(),
});

// Output schema for structured responses
const SearchResultSchema = z.object({
  title: z.string(),
  url: z.string().url(),
  score: z.number().min(0).max(1),
});

server.tool(
  "search",
  "Search with advanced filtering",
  SearchInputSchema,
  async ({ query, limit, filters }) => {
    const results = await performSearch(query, limit, filters);
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(results, null, 2),
        },
      ],
      structuredContent: results,
    };
  }
);
```

### 2.5 Resource Exposure

**File Resources:**
```python
@mcp.resource("file:///{path}")
async def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    with open(path, "r") as f:
        return f.read()
```

**Database Resources:**
```python
@mcp.resource("db://tables/{table}")
async def read_table_schema(table: str) -> str:
    """Get schema for a database table."""
    schema = await db.get_schema(table)
    return json.dumps(schema, indent=2)
```

**API Resources:**
```python
@mcp.resource("api://users/{user_id}")
async def get_user(user_id: str) -> str:
    """Get user data from API."""
    user = await api_client.get_user(user_id)
    return json.dumps(user, indent=2)
```

### 2.6 Prompt Templates

```python
@mcp.prompt()
def code_review(language: str, code: str) -> list[dict]:
    """Generate a code review prompt."""
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"""Please review this {language} code:

```{language}
{code}
```

Analyze for:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security vulnerabilities
5. Suggestions for improvement
"""
            }
        }
    ]


@mcp.prompt()
def data_analysis(dataset: str, question: str) -> list[dict]:
    """Generate a data analysis prompt."""
    return [
        {
            "role": "system",
            "content": {
                "type": "text",
                "text": "You are a data analyst. Analyze the provided dataset and answer questions accurately."
            }
        },
        {
            "role": "user",
            "content": {
                "type": "resource",
                "resource": {
                    "uri": f"data://{dataset}",
                    "mimeType": "text/csv"
                }
            }
        },
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": question
            }
        }
    ]
```

---

## 3. Advanced Patterns

### 3.1 Dynamic Tool Discovery

Tools can be registered and unregistered at runtime:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dynamic-server")
registered_tools = {}

async def register_tool_dynamically(name: str, handler, schema: dict):
    """Register a tool at runtime."""
    registered_tools[name] = {
        "handler": handler,
        "schema": schema
    }
    # Notify clients of tool list change
    await mcp.notify_tools_changed()

async def unregister_tool(name: str):
    """Remove a tool at runtime."""
    if name in registered_tools:
        del registered_tools[name]
        await mcp.notify_tools_changed()

# Override tools/list to include dynamic tools
@mcp.list_tools()
async def list_tools():
    tools = []
    for name, config in registered_tools.items():
        tools.append({
            "name": name,
            "description": config["schema"].get("description", ""),
            "inputSchema": config["schema"]
        })
    return tools

# Override tools/call to handle dynamic tools
@mcp.call_tool()
async def call_tool(name: str, arguments: dict):
    if name in registered_tools:
        handler = registered_tools[name]["handler"]
        return await handler(arguments)
    raise ValueError(f"Unknown tool: {name}")
```

### 3.2 Tool Composition

Combine multiple tools into higher-level operations:

```python
from typing import Any

@mcp.tool()
async def research_topic(topic: str) -> str:
    """Research a topic using multiple tools.

    Composes: web_search, summarize, extract_key_points
    """
    # Step 1: Search for information
    search_results = await web_search(topic)

    # Step 2: Summarize each result
    summaries = []
    for result in search_results[:5]:
        content = await fetch_page(result["url"])
        summary = await summarize(content)
        summaries.append(summary)

    # Step 3: Extract key points
    combined = "\n\n".join(summaries)
    key_points = await extract_key_points(combined)

    return f"""# Research Results for: {topic}

## Key Points
{key_points}

## Sources
{chr(10).join(f"- {r['title']}: {r['url']}" for r in search_results[:5])}
"""

@mcp.tool()
async def pipeline_execute(
    steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Execute a pipeline of tool calls.

    Args:
        steps: List of {tool: str, args: dict} steps
    """
    results = []
    context = {}  # Pass data between steps

    for step in steps:
        tool_name = step["tool"]
        args = step["args"]

        # Template substitution from previous results
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                ref = value[1:]
                args[key] = context.get(ref, value)

        result = await call_tool(tool_name, args)
        results.append({"tool": tool_name, "result": result})
        context[f"step_{len(results)}"] = result

    return results
```

### 3.3 Server Federation

Connect to and aggregate multiple MCP servers:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

class FederatedMCPServer:
    def __init__(self):
        self.servers: dict[str, ClientSession] = {}
        self.tool_registry: dict[str, str] = {}  # tool -> server mapping

    async def connect_server(self, name: str, command: str, args: list[str]):
        """Connect to a downstream MCP server."""
        params = StdioServerParameters(command=command, args=args)
        transport = await stdio_client(params)
        session = ClientSession(*transport)
        await session.initialize()

        # Register all tools from this server
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            prefixed_name = f"{name}_{tool.name}"
            self.tool_registry[prefixed_name] = name

        self.servers[name] = session

    async def list_all_tools(self):
        """List tools from all connected servers."""
        all_tools = []
        for server_name, session in self.servers.items():
            tools = await session.list_tools()
            for tool in tools.tools:
                all_tools.append({
                    "name": f"{server_name}_{tool.name}",
                    "description": f"[{server_name}] {tool.description}",
                    "inputSchema": tool.inputSchema
                })
        return all_tools

    async def call_tool(self, name: str, arguments: dict):
        """Route tool call to appropriate server."""
        if name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {name}")

        server_name = self.tool_registry[name]
        session = self.servers[server_name]

        # Remove server prefix from tool name
        actual_tool_name = name.replace(f"{server_name}_", "", 1)

        return await session.call_tool(actual_tool_name, arguments)

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for session in self.servers.values():
            await session.close()
        self.servers.clear()
        self.tool_registry.clear()
```

### 3.4 Authentication Patterns

**API Key Authentication:**
```python
import os
from functools import wraps

def require_auth(func):
    """Decorator to require API key authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        api_key = os.environ.get("MCP_API_KEY")
        if not api_key:
            raise PermissionError("API key not configured")

        # Validate key format
        if not api_key.startswith("mcp_"):
            raise PermissionError("Invalid API key format")

        return await func(*args, **kwargs)
    return wrapper

@mcp.tool()
@require_auth
async def sensitive_operation(data: str) -> str:
    """A tool requiring authentication."""
    return await process_sensitive_data(data)
```

**OAuth 2.1 for HTTP Transport:**
```python
from mcp.server.fastmcp import FastMCP
from mcp.server.auth import OAuth2Config

oauth_config = OAuth2Config(
    issuer="https://auth.example.com",
    client_id="mcp-server",
    audience="https://mcp.example.com",
    scopes_required=["tools:read", "tools:execute"]
)

mcp = FastMCP("secure-server", auth=oauth_config)

@mcp.tool()
async def protected_tool(ctx: Context, data: str) -> str:
    """Tool that checks OAuth scopes."""
    # Access token claims from context
    claims = ctx.auth_claims
    if "admin" not in claims.get("roles", []):
        raise PermissionError("Admin role required")

    return await admin_operation(data)
```

### 3.5 Rate Limiting

```python
import time
from collections import defaultdict
from functools import wraps

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[key] = [
            t for t in self.requests[key] if t > window_start
        ]

        if len(self.requests[key]) >= self.max_requests:
            return False

        self.requests[key].append(now)
        return True

    def get_retry_after(self, key: str) -> float:
        if not self.requests[key]:
            return 0
        oldest = min(self.requests[key])
        return max(0, oldest + self.window_seconds - time.time())

# Global rate limiter: 100 requests per minute
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

def rate_limited(func):
    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs):
        client_id = ctx.client_id or "anonymous"

        if not rate_limiter.is_allowed(client_id):
            retry_after = rate_limiter.get_retry_after(client_id)
            raise Exception(
                f"Rate limit exceeded. Retry after {retry_after:.0f} seconds"
            )

        return await func(ctx, *args, **kwargs)
    return wrapper

@mcp.tool()
@rate_limited
async def rate_limited_tool(ctx: Context, query: str) -> str:
    """A rate-limited tool."""
    return await process_query(query)
```

---

## 4. Integration Patterns

### 4.1 Claude Desktop Integration

**Configuration file locations:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Basic Configuration:**
```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/weather-server",
        "run",
        "weather_server.py"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Documents",
        "/Users/username/Projects"
      ]
    },
    "database": {
      "command": "python",
      "args": ["/path/to/db_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/mydb"
      }
    }
  }
}
```

**Configuration with Environment Variables:**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### 4.2 Cursor IDE Integration

Cursor supports MCP servers through its settings:

```json
// .cursor/mcp.json in your project
{
  "servers": {
    "project-tools": {
      "command": "node",
      "args": ["./mcp/server.js"],
      "cwd": "${workspaceFolder}"
    },
    "documentation": {
      "command": "python",
      "args": ["-m", "doc_server"],
      "env": {
        "DOC_PATH": "${workspaceFolder}/docs"
      }
    }
  }
}
```

### 4.3 Custom Client Integration

**Python Client:**
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic

class MCPClient:
    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.anthropic = Anthropic()

    async def connect(self, name: str, command: str, args: list[str]):
        """Connect to an MCP server."""
        params = StdioServerParameters(command=command, args=args)
        transport = await stdio_client(params)
        session = ClientSession(*transport)
        await session.initialize()
        self.sessions[name] = session
        print(f"Connected to {name}")

    async def get_all_tools(self) -> list[dict]:
        """Get tools from all connected servers."""
        tools = []
        for name, session in self.sessions.items():
            response = await session.list_tools()
            for tool in response.tools:
                tools.append({
                    "name": f"{name}:{tool.name}",
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        return tools

    async def call_tool(self, full_name: str, arguments: dict):
        """Call a tool on a specific server."""
        server_name, tool_name = full_name.split(":", 1)
        session = self.sessions[server_name]
        return await session.call_tool(tool_name, arguments)

    async def process_with_claude(self, query: str) -> str:
        """Process a query using Claude with MCP tools."""
        tools = await self.get_all_tools()

        # Convert to Anthropic tool format
        anthropic_tools = [
            {
                "name": tool["name"].replace(":", "_"),
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
            for tool in tools
        ]

        messages = [{"role": "user", "content": query}]

        while True:
            response = self.anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=anthropic_tools,
                messages=messages
            )

            # Check for tool use
            tool_uses = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_uses:
                # No tool calls, return text response
                text_blocks = [
                    block.text for block in response.content
                    if block.type == "text"
                ]
                return "\n".join(text_blocks)

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_use in tool_uses:
                # Convert back to server:tool format
                full_name = tool_use.name.replace("_", ":", 1)
                result = await self.call_tool(full_name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result.content)
                })

            messages.append({"role": "user", "content": tool_results})

    async def close(self):
        """Close all connections."""
        for session in self.sessions.values():
            await session.close()


async def main():
    client = MCPClient()

    # Connect to servers
    await client.connect(
        "weather",
        "python",
        ["weather_server.py"]
    )
    await client.connect(
        "filesystem",
        "npx",
        ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    # Process queries
    result = await client.process_with_claude(
        "What's the weather in San Francisco and list files in /tmp?"
    )
    print(result)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.4 Multi-Server Orchestration

```python
class MCPOrchestrator:
    """Orchestrate multiple MCP servers with intelligent routing."""

    def __init__(self):
        self.servers: dict[str, ServerConfig] = {}
        self.tool_capabilities: dict[str, list[str]] = {}

    async def register_server(
        self,
        name: str,
        command: str,
        args: list[str],
        capabilities: list[str]
    ):
        """Register a server with its capabilities."""
        # Connect and store
        session = await self._connect(command, args)
        self.servers[name] = ServerConfig(
            session=session,
            capabilities=capabilities,
            health_check_interval=30
        )

        # Index tools by capability
        for cap in capabilities:
            if cap not in self.tool_capabilities:
                self.tool_capabilities[cap] = []
            self.tool_capabilities[cap].append(name)

    async def route_request(self, capability: str, request: dict):
        """Route request to appropriate server based on capability."""
        if capability not in self.tool_capabilities:
            raise ValueError(f"No server with capability: {capability}")

        # Get healthy servers for this capability
        available = [
            name for name in self.tool_capabilities[capability]
            if self.servers[name].is_healthy
        ]

        if not available:
            raise Exception(f"No healthy servers for: {capability}")

        # Load balance (round-robin)
        server_name = self._select_server(available)
        session = self.servers[server_name].session

        return await session.call_tool(request["tool"], request["arguments"])

    async def execute_workflow(self, workflow: list[dict]):
        """Execute a multi-server workflow."""
        results = []
        context = {}

        for step in workflow:
            capability = step["capability"]
            tool = step["tool"]
            args = self._resolve_args(step["args"], context)

            result = await self.route_request(
                capability,
                {"tool": tool, "arguments": args}
            )

            results.append({
                "step": step["name"],
                "result": result
            })
            context[step["name"]] = result

        return results
```

### 4.5 Fallback Strategies

```python
class FallbackMCPClient:
    """MCP client with fallback support."""

    def __init__(self):
        self.primary_servers: dict[str, ClientSession] = {}
        self.fallback_servers: dict[str, list[ClientSession]] = {}

    async def call_with_fallback(
        self,
        tool: str,
        arguments: dict,
        max_retries: int = 3
    ):
        """Call tool with automatic fallback on failure."""
        server_name = self._get_server_for_tool(tool)

        # Try primary
        try:
            session = self.primary_servers[server_name]
            return await asyncio.wait_for(
                session.call_tool(tool, arguments),
                timeout=30.0
            )
        except Exception as primary_error:
            print(f"Primary failed: {primary_error}")

        # Try fallbacks
        for fallback in self.fallback_servers.get(server_name, []):
            try:
                return await asyncio.wait_for(
                    fallback.call_tool(tool, arguments),
                    timeout=30.0
                )
            except Exception as fallback_error:
                print(f"Fallback failed: {fallback_error}")
                continue

        raise Exception(f"All servers failed for tool: {tool}")

    async def call_with_circuit_breaker(
        self,
        tool: str,
        arguments: dict
    ):
        """Call with circuit breaker pattern."""
        server_name = self._get_server_for_tool(tool)
        breaker = self.circuit_breakers[server_name]

        if breaker.is_open:
            if breaker.should_attempt_reset():
                breaker.half_open()
            else:
                # Use fallback immediately
                return await self._call_fallback(tool, arguments)

        try:
            result = await self.call_with_fallback(tool, arguments)
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
```

---

## 5. Security Best Practices

### 5.1 Permission Scoping (Least Privilege)

```python
from enum import Enum
from typing import Set

class Permission(Enum):
    READ_FILES = "files:read"
    WRITE_FILES = "files:write"
    EXECUTE_COMMANDS = "commands:execute"
    NETWORK_ACCESS = "network:access"
    DATABASE_READ = "database:read"
    DATABASE_WRITE = "database:write"

class PermissionManager:
    def __init__(self):
        self.tool_permissions: dict[str, Set[Permission]] = {}
        self.client_permissions: dict[str, Set[Permission]] = {}

    def register_tool(self, name: str, required_permissions: Set[Permission]):
        """Register required permissions for a tool."""
        self.tool_permissions[name] = required_permissions

    def grant_client_permissions(
        self,
        client_id: str,
        permissions: Set[Permission]
    ):
        """Grant permissions to a client."""
        self.client_permissions[client_id] = permissions

    def check_permission(
        self,
        client_id: str,
        tool_name: str
    ) -> tuple[bool, Set[Permission]]:
        """Check if client can use tool, return missing permissions."""
        required = self.tool_permissions.get(tool_name, set())
        granted = self.client_permissions.get(client_id, set())
        missing = required - granted
        return len(missing) == 0, missing

# Usage
perm_manager = PermissionManager()

# Register tools with required permissions
perm_manager.register_tool(
    "read_file",
    {Permission.READ_FILES}
)
perm_manager.register_tool(
    "write_file",
    {Permission.READ_FILES, Permission.WRITE_FILES}
)
perm_manager.register_tool(
    "execute_query",
    {Permission.DATABASE_READ}
)

@mcp.tool()
async def read_file(ctx: Context, path: str) -> str:
    """Read a file with permission checking."""
    client_id = ctx.client_id
    allowed, missing = perm_manager.check_permission(client_id, "read_file")

    if not allowed:
        raise PermissionError(
            f"Missing permissions: {[p.value for p in missing]}"
        )

    return await safe_read_file(path)
```

### 5.2 Input Validation

```python
from pydantic import BaseModel, Field, validator
from pathlib import Path
import re

class FileReadInput(BaseModel):
    path: str = Field(..., description="File path to read")

    @validator("path")
    def validate_path(cls, v):
        # Prevent path traversal
        path = Path(v).resolve()
        allowed_dirs = [Path("/allowed/dir1"), Path("/allowed/dir2")]

        if not any(
            path.is_relative_to(allowed) for allowed in allowed_dirs
        ):
            raise ValueError(f"Access denied: {path}")

        # Block sensitive files
        blocked_patterns = [
            r"\.env$",
            r"\.git/",
            r"credentials",
            r"secrets?\.json$",
            r"\.ssh/"
        ]

        for pattern in blocked_patterns:
            if re.search(pattern, str(path), re.IGNORECASE):
                raise ValueError(f"Access to sensitive file denied: {path}")

        return str(path)

class SQLQueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")

    @validator("query")
    def validate_query(cls, v):
        # Only allow SELECT statements
        normalized = v.strip().upper()
        if not normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Block dangerous keywords
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        for keyword in dangerous:
            if keyword in normalized:
                raise ValueError(f"Dangerous keyword detected: {keyword}")

        return v

@mcp.tool()
async def read_file(input: FileReadInput) -> str:
    """Read file with validated input."""
    # Input already validated by Pydantic
    with open(input.path, "r") as f:
        return f.read()

@mcp.tool()
async def query_database(input: SQLQueryInput) -> str:
    """Execute validated SQL query."""
    # Query already validated
    results = await db.execute(input.query)
    return json.dumps(results, indent=2)
```

### 5.3 Output Sanitization

```python
import re
from typing import Any

class OutputSanitizer:
    """Sanitize tool outputs before returning to clients."""

    PATTERNS = {
        "api_key": r"(?:api[_-]?key|apikey)[\"']?\s*[:=]\s*[\"']?([a-zA-Z0-9_\-]{20,})",
        "password": r"(?:password|passwd|pwd)[\"']?\s*[:=]\s*[\"']?([^\s\"']+)",
        "token": r"(?:bearer|token|auth)[\"']?\s*[:=]?\s*[\"']?([a-zA-Z0-9_\-\.]{20,})",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "private_key": r"-----BEGIN (?:RSA )?PRIVATE KEY-----",
    }

    def __init__(self, redaction_string: str = "[REDACTED]"):
        self.redaction = redaction_string

    def sanitize(self, content: str) -> str:
        """Remove sensitive information from content."""
        result = content

        for name, pattern in self.PATTERNS.items():
            result = re.sub(
                pattern,
                f"{name.upper()}={self.redaction}",
                result,
                flags=re.IGNORECASE
            )

        return result

    def sanitize_dict(self, data: dict) -> dict:
        """Recursively sanitize dictionary values."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.sanitize(value)
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.sanitize_dict(v) if isinstance(v, dict)
                    else self.sanitize(v) if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

sanitizer = OutputSanitizer()

@mcp.tool()
async def search_logs(query: str) -> str:
    """Search logs with output sanitization."""
    raw_results = await log_search(query)
    sanitized = sanitizer.sanitize(raw_results)
    return sanitized
```

### 5.4 Audit Logging

```python
import logging
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    timestamp: str
    event_type: str
    client_id: str
    tool_name: Optional[str]
    resource_uri: Optional[str]
    action: str
    result: str
    details: dict

class AuditLogger:
    def __init__(self, log_path: str = "/var/log/mcp/audit.log"):
        self.logger = logging.getLogger("mcp.audit")
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def log_event(self, event: AuditEvent):
        self.logger.info(json.dumps(asdict(event)))

    def log_tool_call(
        self,
        client_id: str,
        tool_name: str,
        arguments: dict,
        result: str,
        success: bool
    ):
        self.log_event(AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="tool_call",
            client_id=client_id,
            tool_name=tool_name,
            resource_uri=None,
            action="execute",
            result="success" if success else "failure",
            details={
                "arguments": arguments,
                "response_preview": result[:500] if result else None
            }
        ))

    def log_resource_access(
        self,
        client_id: str,
        resource_uri: str,
        success: bool
    ):
        self.log_event(AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type="resource_access",
            client_id=client_id,
            tool_name=None,
            resource_uri=resource_uri,
            action="read",
            result="success" if success else "failure",
            details={}
        ))

audit = AuditLogger()

@mcp.tool()
async def audited_tool(ctx: Context, data: str) -> str:
    """Tool with audit logging."""
    client_id = ctx.client_id or "anonymous"

    try:
        result = await process_data(data)
        audit.log_tool_call(
            client_id=client_id,
            tool_name="audited_tool",
            arguments={"data": data[:100]},  # Truncate for logging
            result=result[:500],
            success=True
        )
        return result
    except Exception as e:
        audit.log_tool_call(
            client_id=client_id,
            tool_name="audited_tool",
            arguments={"data": data[:100]},
            result=str(e),
            success=False
        )
        raise
```

### 5.5 Secrets Management

```python
import os
from typing import Optional
from functools import lru_cache

class SecretsManager:
    """Manage secrets without exposing them in code or config."""

    def __init__(self):
        self._secrets_cache: dict[str, str] = {}

    @lru_cache(maxsize=100)
    def get_secret(self, name: str) -> Optional[str]:
        """Get secret from environment or secrets manager."""
        # 1. Try environment variable
        env_value = os.environ.get(name)
        if env_value:
            return env_value

        # 2. Try secrets file (for development)
        secrets_file = os.environ.get("SECRETS_FILE", ".secrets")
        if os.path.exists(secrets_file):
            with open(secrets_file) as f:
                for line in f:
                    if line.startswith(f"{name}="):
                        return line.split("=", 1)[1].strip()

        # 3. Try cloud secrets manager (production)
        # Example: AWS Secrets Manager, HashiCorp Vault
        try:
            return self._fetch_from_vault(name)
        except Exception:
            pass

        return None

    def _fetch_from_vault(self, name: str) -> Optional[str]:
        """Fetch secret from HashiCorp Vault."""
        vault_addr = os.environ.get("VAULT_ADDR")
        vault_token = os.environ.get("VAULT_TOKEN")

        if not vault_addr or not vault_token:
            return None

        import hvac
        client = hvac.Client(url=vault_addr, token=vault_token)
        secret = client.secrets.kv.v2.read_secret_version(path=name)
        return secret["data"]["data"].get("value")

    def require_secret(self, name: str) -> str:
        """Get secret or raise if not found."""
        value = self.get_secret(name)
        if value is None:
            raise ValueError(f"Required secret not found: {name}")
        return value

secrets = SecretsManager()

# Usage in server
@mcp.tool()
async def call_external_api(endpoint: str) -> str:
    """Call external API with managed secrets."""
    api_key = secrets.require_secret("EXTERNAL_API_KEY")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        return response.text
```

---

## 6. Production Deployment

### 6.1 Server Hosting Options

| Option | Use Case | Pros | Cons |
|--------|----------|------|------|
| **stdio (local)** | Desktop apps, CLI tools | Simple, secure, no network | Single client only |
| **Docker Container** | Microservices, cloud | Isolated, portable, scalable | Added complexity |
| **Kubernetes** | Enterprise, multi-tenant | Auto-scaling, HA | Operational overhead |
| **Serverless** | Event-driven, low traffic | Cost-effective, scalable | Cold starts, limits |
| **Dedicated VM** | High-performance, single tenant | Full control | Higher cost |

**Docker Deployment:**
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Create non-root user
RUN useradd -m mcpuser
USER mcpuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

# Run with HTTP transport
CMD ["python", "-m", "mcp_server", "--transport", "http", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MCP_LOG_LEVEL=info
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./data:/app/data:ro
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 6.2 Scaling Considerations

```python
# Horizontal scaling with load balancing
from mcp.server.fastmcp import FastMCP
from mcp.server.http import StreamableHTTPTransport

mcp = FastMCP("scalable-server")

# Configure for stateless operation
@mcp.tool()
async def stateless_tool(data: str) -> str:
    """Stateless tool that can run on any instance."""
    # Use external state stores
    cache = await redis.get(f"cache:{data}")
    if cache:
        return cache

    result = await expensive_computation(data)
    await redis.set(f"cache:{data}", result, ex=3600)
    return result

# Run with HTTP transport for load balancing
if __name__ == "__main__":
    transport = StreamableHTTPTransport(
        host="0.0.0.0",
        port=8080,
        cors_origins=["https://allowed-origin.com"]
    )
    mcp.run(transport=transport)
```

**Kubernetes Deployment:**
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: mcp-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        env:
        - name: MCP_INSTANCE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 6.3 Monitoring and Observability

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
TOOL_CALLS = Counter(
    "mcp_tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)
TOOL_LATENCY = Histogram(
    "mcp_tool_latency_seconds",
    "Tool call latency",
    ["tool_name"],
    buckets=[.01, .05, .1, .25, .5, 1, 2.5, 5, 10]
)
ACTIVE_CONNECTIONS = Gauge(
    "mcp_active_connections",
    "Number of active client connections"
)

def instrument_tool(func):
    """Decorator to add metrics to tools."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            TOOL_CALLS.labels(tool_name=tool_name, status="success").inc()
            return result
        except Exception as e:
            TOOL_CALLS.labels(tool_name=tool_name, status="error").inc()
            raise
        finally:
            duration = time.time() - start_time
            TOOL_LATENCY.labels(tool_name=tool_name).observe(duration)

    return wrapper

# Start metrics server
start_http_server(9090)

@mcp.tool()
@instrument_tool
async def monitored_tool(data: str) -> str:
    """Tool with Prometheus metrics."""
    return await process_data(data)
```

**Structured Logging:**
```python
import structlog
import json

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

@mcp.tool()
async def logged_tool(ctx: Context, query: str) -> str:
    """Tool with structured logging."""
    log = logger.bind(
        tool="logged_tool",
        client_id=ctx.client_id,
        request_id=ctx.request_id
    )

    log.info("tool_invoked", query_length=len(query))

    try:
        result = await process_query(query)
        log.info("tool_completed", result_length=len(result))
        return result
    except Exception as e:
        log.error("tool_failed", error=str(e))
        raise
```

### 6.4 Versioning Strategies

```python
from enum import Enum

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"

# Version-aware server
mcp_v1 = FastMCP("my-server-v1", version="1.0.0")
mcp_v2 = FastMCP("my-server-v2", version="2.0.0")

# V1 tool (original)
@mcp_v1.tool()
async def get_data(id: str) -> str:
    """Get data by ID (v1)."""
    return await legacy_get_data(id)

# V2 tool (enhanced)
@mcp_v2.tool()
async def get_data(
    id: str,
    include_metadata: bool = False,
    format: str = "json"
) -> str:
    """Get data by ID with options (v2)."""
    data = await enhanced_get_data(id)
    if include_metadata:
        data["_metadata"] = await get_metadata(id)
    if format == "yaml":
        return yaml.dump(data)
    return json.dumps(data)

# Version routing
async def route_request(version: str, tool: str, args: dict):
    if version == "v1":
        return await mcp_v1.call_tool(tool, args)
    elif version == "v2":
        return await mcp_v2.call_tool(tool, args)
    else:
        raise ValueError(f"Unknown version: {version}")
```

### 6.5 Health Checks

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class HealthStatus:
    healthy: bool
    message: str
    last_check: datetime
    details: dict

class HealthChecker:
    def __init__(self):
        self.checks: dict[str, callable] = {}
        self.last_results: dict[str, HealthStatus] = {}

    def register_check(self, name: str, check_func: callable):
        self.checks[name] = check_func

    async def run_checks(self) -> dict:
        results = {}
        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                await check_func()
                results[name] = HealthStatus(
                    healthy=True,
                    message="OK",
                    last_check=datetime.utcnow(),
                    details={}
                )
            except Exception as e:
                all_healthy = False
                results[name] = HealthStatus(
                    healthy=False,
                    message=str(e),
                    last_check=datetime.utcnow(),
                    details={"error": type(e).__name__}
                )

        self.last_results = results
        return {
            "healthy": all_healthy,
            "checks": {
                name: {
                    "healthy": status.healthy,
                    "message": status.message,
                    "last_check": status.last_check.isoformat()
                }
                for name, status in results.items()
            }
        }

health = HealthChecker()

# Register health checks
async def check_database():
    await db.execute("SELECT 1")

async def check_redis():
    await redis.ping()

async def check_external_api():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.example.com/health",
            timeout=5.0
        )
        response.raise_for_status()

health.register_check("database", check_database)
health.register_check("redis", check_redis)
health.register_check("external_api", check_external_api)

# HTTP endpoints for health checks
@app.get("/health")
async def health_endpoint():
    """Liveness probe - is the server running?"""
    return {"status": "alive"}

@app.get("/ready")
async def ready_endpoint():
    """Readiness probe - can the server handle requests?"""
    result = await health.run_checks()
    if result["healthy"]:
        return result
    raise HTTPException(status_code=503, detail=result)
```

---

## 7. Complete Production Examples

### 7.1 Filesystem MCP Server

```python
# filesystem_server.py
"""
Production-ready filesystem MCP server with security controls.
"""

import os
import stat
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, validator

# Configuration
ALLOWED_DIRECTORIES = [
    Path(os.environ.get("MCP_ALLOWED_DIR_1", "/tmp")),
    Path(os.environ.get("MCP_ALLOWED_DIR_2", "/home/user/documents")),
]
MAX_FILE_SIZE = int(os.environ.get("MCP_MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
BLOCKED_EXTENSIONS = {".exe", ".dll", ".so", ".sh", ".bat", ".cmd"}

mcp = FastMCP("filesystem")


class PathValidator:
    """Validate and sanitize file paths."""

    @staticmethod
    def is_allowed(path: Path) -> bool:
        resolved = path.resolve()
        return any(
            resolved.is_relative_to(allowed.resolve())
            for allowed in ALLOWED_DIRECTORIES
        )

    @staticmethod
    def is_safe_extension(path: Path) -> bool:
        return path.suffix.lower() not in BLOCKED_EXTENSIONS

    @staticmethod
    def validate(path_str: str) -> Path:
        path = Path(path_str).resolve()

        if not PathValidator.is_allowed(path):
            raise PermissionError(
                f"Access denied: {path} is outside allowed directories"
            )

        if not PathValidator.is_safe_extension(path):
            raise PermissionError(
                f"Blocked file extension: {path.suffix}"
            )

        return path


class FileInfo(BaseModel):
    path: str
    name: str
    size: int
    is_directory: bool
    modified: str
    permissions: str


@mcp.tool()
async def list_directory(
    path: str,
    include_hidden: bool = False
) -> list[FileInfo]:
    """List contents of a directory.

    Args:
        path: Directory path to list
        include_hidden: Include hidden files (starting with .)
    """
    dir_path = PathValidator.validate(path)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {path}")

    entries = []
    for entry in dir_path.iterdir():
        if not include_hidden and entry.name.startswith("."):
            continue

        stat_info = entry.stat()
        entries.append(FileInfo(
            path=str(entry),
            name=entry.name,
            size=stat_info.st_size,
            is_directory=entry.is_dir(),
            modified=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            permissions=stat.filemode(stat_info.st_mode)
        ))

    return sorted(entries, key=lambda x: (not x.is_directory, x.name.lower()))


@mcp.tool()
async def read_file(
    path: str,
    encoding: str = "utf-8",
    max_lines: Optional[int] = None
) -> str:
    """Read contents of a file.

    Args:
        path: File path to read
        encoding: Text encoding (default: utf-8)
        max_lines: Maximum lines to read (None for all)
    """
    file_path = PathValidator.validate(path)

    if not file_path.is_file():
        raise ValueError(f"Not a file: {path}")

    if file_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {file_path.stat().st_size} bytes "
            f"(max: {MAX_FILE_SIZE})"
        )

    with open(file_path, "r", encoding=encoding) as f:
        if max_lines:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
            return "".join(lines)
        return f.read()


@mcp.tool()
async def write_file(
    path: str,
    content: str,
    create_dirs: bool = False
) -> dict:
    """Write content to a file.

    Args:
        path: File path to write
        content: Content to write
        create_dirs: Create parent directories if needed
    """
    file_path = PathValidator.validate(path)

    if len(content.encode("utf-8")) > MAX_FILE_SIZE:
        raise ValueError(f"Content too large (max: {MAX_FILE_SIZE} bytes)")

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "path": str(file_path),
        "size": len(content.encode("utf-8")),
        "hash": hashlib.sha256(content.encode()).hexdigest()
    }


@mcp.tool()
async def search_files(
    directory: str,
    pattern: str,
    recursive: bool = True,
    max_results: int = 100
) -> list[str]:
    """Search for files matching a pattern.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "*.py", "**/*.txt")
        recursive: Search subdirectories
        max_results: Maximum results to return
    """
    dir_path = PathValidator.validate(directory)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    if recursive and not pattern.startswith("**"):
        pattern = f"**/{pattern}"

    results = []
    for match in dir_path.glob(pattern):
        if PathValidator.is_allowed(match):
            results.append(str(match))
            if len(results) >= max_results:
                break

    return results


@mcp.resource("file:///{path}")
async def file_resource(path: str) -> str:
    """Read file as a resource."""
    return await read_file(path)


def main():
    print(f"Filesystem server starting...")
    print(f"Allowed directories: {ALLOWED_DIRECTORIES}")
    print(f"Max file size: {MAX_FILE_SIZE} bytes")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

### 7.2 Database Query MCP Server

```python
# database_server.py
"""
Production-ready database MCP server with query safety.
"""

import os
import re
from typing import Any, Optional
from datetime import datetime

import asyncpg
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel

# Configuration
DATABASE_URL = os.environ["DATABASE_URL"]
MAX_ROWS = int(os.environ.get("MCP_MAX_ROWS", 1000))
QUERY_TIMEOUT = int(os.environ.get("MCP_QUERY_TIMEOUT", 30))
ALLOWED_SCHEMAS = os.environ.get("MCP_ALLOWED_SCHEMAS", "public").split(",")

mcp = FastMCP("database")
pool: Optional[asyncpg.Pool] = None


class QueryValidator:
    """Validate SQL queries for safety."""

    DANGEROUS_PATTERNS = [
        r"\bDROP\b",
        r"\bDELETE\b",
        r"\bTRUNCATE\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r";\s*\w",  # Multiple statements
        r"--",     # SQL comments
        r"/\*",    # Block comments
    ]

    @classmethod
    def validate(cls, query: str) -> str:
        """Validate and sanitize query."""
        normalized = query.strip().upper()

        # Must start with SELECT
        if not normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, normalized):
                raise ValueError(f"Dangerous SQL pattern detected")

        return query.strip()


class TableInfo(BaseModel):
    schema: str
    name: str
    columns: list[dict]
    row_count: Optional[int]


@mcp.on_startup()
async def startup():
    """Initialize database connection pool."""
    global pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=QUERY_TIMEOUT
    )


@mcp.on_shutdown()
async def shutdown():
    """Close database connection pool."""
    global pool
    if pool:
        await pool.close()


@mcp.tool()
async def list_tables(schema: str = "public") -> list[TableInfo]:
    """List all tables in a schema.

    Args:
        schema: Database schema (default: public)
    """
    if schema not in ALLOWED_SCHEMAS:
        raise PermissionError(f"Schema not allowed: {schema}")

    query = """
        SELECT
            t.table_schema,
            t.table_name,
            (SELECT json_agg(json_build_object(
                'name', c.column_name,
                'type', c.data_type,
                'nullable', c.is_nullable
            ))
            FROM information_schema.columns c
            WHERE c.table_schema = t.table_schema
              AND c.table_name = t.table_name
            ) as columns
        FROM information_schema.tables t
        WHERE t.table_schema = $1
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, schema)
        return [
            TableInfo(
                schema=row["table_schema"],
                name=row["table_name"],
                columns=row["columns"] or [],
                row_count=None
            )
            for row in rows
        ]


@mcp.tool()
async def execute_query(
    query: str,
    params: Optional[list[Any]] = None,
    limit: int = 100
) -> dict:
    """Execute a SELECT query and return results.

    Args:
        query: SQL SELECT query
        params: Query parameters for prepared statement
        limit: Maximum rows to return (max: 1000)
    """
    validated_query = QueryValidator.validate(query)
    limit = min(limit, MAX_ROWS)

    # Add LIMIT if not present
    if "LIMIT" not in validated_query.upper():
        validated_query = f"{validated_query} LIMIT {limit}"

    async with pool.acquire() as conn:
        start_time = datetime.now()

        if params:
            rows = await conn.fetch(validated_query, *params)
        else:
            rows = await conn.fetch(validated_query)

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "columns": list(rows[0].keys()) if rows else [],
            "rows": [dict(row) for row in rows],
            "row_count": len(rows),
            "execution_time_seconds": execution_time,
            "truncated": len(rows) >= limit
        }


@mcp.tool()
async def get_table_sample(
    table: str,
    schema: str = "public",
    sample_size: int = 10
) -> dict:
    """Get a sample of rows from a table.

    Args:
        table: Table name
        schema: Schema name (default: public)
        sample_size: Number of sample rows (max: 100)
    """
    if schema not in ALLOWED_SCHEMAS:
        raise PermissionError(f"Schema not allowed: {schema}")

    # Validate table name (prevent SQL injection)
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        raise ValueError("Invalid table name")

    sample_size = min(sample_size, 100)

    query = f'SELECT * FROM "{schema}"."{table}" LIMIT {sample_size}'
    return await execute_query(query)


@mcp.tool()
async def analyze_table(
    table: str,
    schema: str = "public"
) -> dict:
    """Analyze table structure and statistics.

    Args:
        table: Table name
        schema: Schema name (default: public)
    """
    if schema not in ALLOWED_SCHEMAS:
        raise PermissionError(f"Schema not allowed: {schema}")

    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table):
        raise ValueError("Invalid table name")

    async with pool.acquire() as conn:
        # Get column info
        columns_query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """
        columns = await conn.fetch(columns_query, schema, table)

        # Get row count
        count_query = f'SELECT COUNT(*) FROM "{schema}"."{table}"'
        count = await conn.fetchval(count_query)

        # Get indexes
        indexes_query = """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = $1 AND tablename = $2
        """
        indexes = await conn.fetch(indexes_query, schema, table)

        return {
            "schema": schema,
            "table": table,
            "row_count": count,
            "columns": [dict(c) for c in columns],
            "indexes": [dict(i) for i in indexes]
        }


@mcp.resource("db://schemas")
async def list_schemas() -> str:
    """List available database schemas."""
    import json
    return json.dumps({
        "allowed_schemas": ALLOWED_SCHEMAS,
        "max_rows": MAX_ROWS,
        "query_timeout": QUERY_TIMEOUT
    })


def main():
    print("Database MCP server starting...")
    print(f"Allowed schemas: {ALLOWED_SCHEMAS}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

### 7.3 API Gateway MCP Server

```python
# api_gateway_server.py
"""
Production-ready API gateway MCP server with caching and rate limiting.
"""

import os
import json
import hashlib
from typing import Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache

import httpx
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, HttpUrl

# Configuration
CACHE_TTL = int(os.environ.get("MCP_CACHE_TTL", 300))
RATE_LIMIT_REQUESTS = int(os.environ.get("MCP_RATE_LIMIT", 100))
RATE_LIMIT_WINDOW = int(os.environ.get("MCP_RATE_WINDOW", 60))

mcp = FastMCP("api-gateway")

# Simple in-memory cache (use Redis in production)
cache: dict[str, tuple[Any, datetime]] = {}
rate_limits: dict[str, list[datetime]] = {}


class APIEndpoint(BaseModel):
    name: str
    base_url: HttpUrl
    auth_type: str = "none"  # none, bearer, api_key
    auth_header: str = "Authorization"
    timeout: int = 30


# Registered API endpoints
ENDPOINTS: dict[str, APIEndpoint] = {
    "github": APIEndpoint(
        name="github",
        base_url="https://api.github.com",
        auth_type="bearer",
        auth_header="Authorization"
    ),
    "weather": APIEndpoint(
        name="weather",
        base_url="https://api.weather.gov",
        auth_type="none"
    ),
    "jsonplaceholder": APIEndpoint(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com",
        auth_type="none"
    ),
}


class RateLimiter:
    @staticmethod
    def check(key: str) -> bool:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

        if key not in rate_limits:
            rate_limits[key] = []

        # Clean old entries
        rate_limits[key] = [t for t in rate_limits[key] if t > window_start]

        if len(rate_limits[key]) >= RATE_LIMIT_REQUESTS:
            return False

        rate_limits[key].append(now)
        return True


class CacheManager:
    @staticmethod
    def get(key: str) -> Optional[Any]:
        if key in cache:
            value, expiry = cache[key]
            if datetime.utcnow() < expiry:
                return value
            del cache[key]
        return None

    @staticmethod
    def set(key: str, value: Any, ttl: int = CACHE_TTL):
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        cache[key] = (value, expiry)

    @staticmethod
    def cache_key(endpoint: str, path: str, params: dict) -> str:
        data = f"{endpoint}:{path}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(data.encode()).hexdigest()


def get_auth_header(endpoint: APIEndpoint) -> dict:
    """Get authentication header for endpoint."""
    if endpoint.auth_type == "none":
        return {}

    env_key = f"{endpoint.name.upper()}_API_KEY"
    api_key = os.environ.get(env_key)

    if not api_key:
        raise ValueError(f"Missing API key: {env_key}")

    if endpoint.auth_type == "bearer":
        return {endpoint.auth_header: f"Bearer {api_key}"}
    elif endpoint.auth_type == "api_key":
        return {endpoint.auth_header: api_key}

    return {}


@mcp.tool()
async def list_endpoints() -> list[dict]:
    """List available API endpoints."""
    return [
        {
            "name": e.name,
            "base_url": str(e.base_url),
            "auth_required": e.auth_type != "none"
        }
        for e in ENDPOINTS.values()
    ]


@mcp.tool()
async def api_get(
    endpoint: str,
    path: str,
    params: Optional[dict] = None,
    use_cache: bool = True
) -> dict:
    """Make a GET request to an API endpoint.

    Args:
        endpoint: API endpoint name (e.g., "github", "weather")
        path: API path (e.g., "/users/octocat")
        params: Query parameters
        use_cache: Use cached response if available
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    if not RateLimiter.check(endpoint):
        raise Exception(f"Rate limit exceeded for {endpoint}")

    ep = ENDPOINTS[endpoint]
    params = params or {}

    # Check cache
    cache_key = CacheManager.cache_key(endpoint, path, params)
    if use_cache:
        cached = CacheManager.get(cache_key)
        if cached:
            return {"data": cached, "cached": True}

    # Make request
    url = f"{ep.base_url}{path}"
    headers = {"User-Agent": "MCP-API-Gateway/1.0"}
    headers.update(get_auth_header(ep))

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params=params,
            headers=headers,
            timeout=ep.timeout
        )
        response.raise_for_status()
        data = response.json()

    # Cache response
    CacheManager.set(cache_key, data)

    return {"data": data, "cached": False}


@mcp.tool()
async def api_post(
    endpoint: str,
    path: str,
    body: dict,
    params: Optional[dict] = None
) -> dict:
    """Make a POST request to an API endpoint.

    Args:
        endpoint: API endpoint name
        path: API path
        body: Request body (JSON)
        params: Query parameters
    """
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    if not RateLimiter.check(endpoint):
        raise Exception(f"Rate limit exceeded for {endpoint}")

    ep = ENDPOINTS[endpoint]

    url = f"{ep.base_url}{path}"
    headers = {
        "User-Agent": "MCP-API-Gateway/1.0",
        "Content-Type": "application/json"
    }
    headers.update(get_auth_header(ep))

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=body,
            params=params or {},
            headers=headers,
            timeout=ep.timeout
        )
        response.raise_for_status()
        return {"data": response.json(), "status": response.status_code}


@mcp.tool()
async def batch_requests(
    requests: list[dict]
) -> list[dict]:
    """Execute multiple API requests in parallel.

    Args:
        requests: List of request configs with endpoint, path, params
    """
    import asyncio

    async def execute_one(req: dict) -> dict:
        try:
            result = await api_get(
                endpoint=req["endpoint"],
                path=req["path"],
                params=req.get("params"),
                use_cache=req.get("use_cache", True)
            )
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*[execute_one(r) for r in requests])
    return list(results)


@mcp.resource("api://endpoints")
async def endpoints_resource() -> str:
    """Get endpoint configuration as resource."""
    endpoints = await list_endpoints()
    return json.dumps(endpoints, indent=2)


def main():
    print("API Gateway MCP server starting...")
    print(f"Registered endpoints: {list(ENDPOINTS.keys())}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

### 7.4 Composite Multi-Tool Server

```python
# composite_server.py
"""
Composite MCP server demonstrating multi-tool coordination.
"""

import os
import json
from typing import Any, Optional
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel

mcp = FastMCP("composite-tools")


# Tool categories for organization
class ToolCategory:
    DATA = "data"
    TRANSFORM = "transform"
    ANALYSIS = "analysis"
    WORKFLOW = "workflow"


# ---------- Data Tools ----------

@mcp.tool()
async def fetch_json(url: str) -> dict:
    """Fetch JSON data from a URL.

    Category: data
    """
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def parse_csv(content: str, has_header: bool = True) -> list[dict]:
    """Parse CSV content into structured data.

    Category: data
    """
    import csv
    from io import StringIO

    reader = csv.reader(StringIO(content))
    rows = list(reader)

    if not rows:
        return []

    if has_header:
        headers = rows[0]
        return [dict(zip(headers, row)) for row in rows[1:]]
    else:
        return [{"col_" + str(i): v for i, v in enumerate(row)} for row in rows]


# ---------- Transform Tools ----------

@mcp.tool()
async def filter_data(
    data: list[dict],
    field: str,
    operator: str,
    value: Any
) -> list[dict]:
    """Filter a list of objects by field condition.

    Category: transform
    Args:
        data: List of dictionaries
        field: Field name to filter on
        operator: Comparison operator (eq, ne, gt, lt, gte, lte, contains)
        value: Value to compare against
    """
    ops = {
        "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
        "contains": lambda a, b: b in str(a),
    }

    if operator not in ops:
        raise ValueError(f"Unknown operator: {operator}")

    op_func = ops[operator]
    return [item for item in data if op_func(item.get(field), value)]


@mcp.tool()
async def map_fields(
    data: list[dict],
    field_mapping: dict[str, str]
) -> list[dict]:
    """Rename fields in a list of objects.

    Category: transform
    Args:
        data: List of dictionaries
        field_mapping: Old field name -> new field name mapping
    """
    result = []
    for item in data:
        new_item = {}
        for old_key, new_key in field_mapping.items():
            if old_key in item:
                new_item[new_key] = item[old_key]
        # Keep unmapped fields
        for key, value in item.items():
            if key not in field_mapping:
                new_item[key] = value
        result.append(new_item)
    return result


@mcp.tool()
async def aggregate_data(
    data: list[dict],
    group_by: str,
    aggregations: dict[str, str]
) -> list[dict]:
    """Aggregate data by grouping field.

    Category: transform
    Args:
        data: List of dictionaries
        group_by: Field to group by
        aggregations: Field -> aggregation type (sum, avg, count, min, max)
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for item in data:
        key = item.get(group_by)
        groups[key].append(item)

    result = []
    for group_key, items in groups.items():
        row = {group_by: group_key}

        for field, agg_type in aggregations.items():
            values = [item.get(field, 0) for item in items if item.get(field) is not None]

            if agg_type == "sum":
                row[f"{field}_sum"] = sum(values)
            elif agg_type == "avg":
                row[f"{field}_avg"] = sum(values) / len(values) if values else 0
            elif agg_type == "count":
                row[f"{field}_count"] = len(values)
            elif agg_type == "min":
                row[f"{field}_min"] = min(values) if values else None
            elif agg_type == "max":
                row[f"{field}_max"] = max(values) if values else None

        result.append(row)

    return result


# ---------- Analysis Tools ----------

@mcp.tool()
async def analyze_structure(data: Any) -> dict:
    """Analyze the structure of data.

    Category: analysis
    """
    def get_type_info(obj, depth=0, max_depth=5):
        if depth > max_depth:
            return {"type": "...", "truncated": True}

        if isinstance(obj, dict):
            return {
                "type": "object",
                "keys": list(obj.keys())[:20],
                "key_count": len(obj),
                "sample": {
                    k: get_type_info(v, depth + 1)
                    for k, v in list(obj.items())[:5]
                }
            }
        elif isinstance(obj, list):
            return {
                "type": "array",
                "length": len(obj),
                "item_type": get_type_info(obj[0], depth + 1) if obj else None
            }
        else:
            return {"type": type(obj).__name__}

    return get_type_info(data)


@mcp.tool()
async def summarize_numeric(
    data: list[dict],
    fields: list[str]
) -> dict:
    """Generate statistical summary of numeric fields.

    Category: analysis
    """
    import statistics

    summaries = {}
    for field in fields:
        values = [
            item[field] for item in data
            if field in item and isinstance(item[field], (int, float))
        ]

        if not values:
            summaries[field] = {"error": "No numeric values"}
            continue

        summaries[field] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }

    return summaries


# ---------- Workflow Tools ----------

@mcp.tool()
async def create_pipeline(
    steps: list[dict]
) -> dict:
    """Create and validate a data processing pipeline.

    Category: workflow
    Args:
        steps: List of pipeline steps with tool and args
    """
    available_tools = {
        "fetch_json", "parse_csv", "filter_data",
        "map_fields", "aggregate_data", "analyze_structure",
        "summarize_numeric"
    }

    validated_steps = []
    for i, step in enumerate(steps):
        tool = step.get("tool")
        if tool not in available_tools:
            raise ValueError(f"Step {i}: Unknown tool '{tool}'")
        validated_steps.append({
            "step": i,
            "tool": tool,
            "args": step.get("args", {}),
            "output_var": step.get("output", f"step_{i}_result")
        })

    return {
        "pipeline_id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "steps": validated_steps,
        "status": "validated"
    }


@mcp.tool()
async def execute_pipeline(
    pipeline: dict,
    initial_data: Optional[Any] = None
) -> dict:
    """Execute a validated data processing pipeline.

    Category: workflow
    """
    tool_functions = {
        "fetch_json": fetch_json,
        "parse_csv": parse_csv,
        "filter_data": filter_data,
        "map_fields": map_fields,
        "aggregate_data": aggregate_data,
        "analyze_structure": analyze_structure,
        "summarize_numeric": summarize_numeric,
    }

    context = {"input": initial_data}
    results = []

    for step in pipeline["steps"]:
        tool_name = step["tool"]
        args = step["args"].copy()

        # Replace variable references
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in context:
                    args[key] = context[var_name]

        # Execute tool
        tool_func = tool_functions[tool_name]
        result = await tool_func(**args)

        # Store result
        output_var = step["output_var"]
        context[output_var] = result
        results.append({
            "step": step["step"],
            "tool": tool_name,
            "output_var": output_var,
            "success": True
        })

    return {
        "pipeline_id": pipeline["pipeline_id"],
        "results": results,
        "final_output": context.get(pipeline["steps"][-1]["output_var"]),
        "status": "completed"
    }


@mcp.prompt()
def data_pipeline_prompt(
    data_source: str,
    objective: str
) -> list[dict]:
    """Generate a prompt for creating a data pipeline."""
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"""Help me create a data processing pipeline.

Data Source: {data_source}
Objective: {objective}

Available tools:
- fetch_json: Fetch JSON from URL
- parse_csv: Parse CSV content
- filter_data: Filter by field condition
- map_fields: Rename fields
- aggregate_data: Group and aggregate
- analyze_structure: Analyze data structure
- summarize_numeric: Statistical summary

Please suggest a pipeline with specific steps and arguments.
"""
            }
        }
    ]


def main():
    print("Composite MCP server starting...")
    print("Available tool categories: data, transform, analysis, workflow")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

---

## Summary

The Model Context Protocol represents a significant standardization effort for agentic AI systems, now stewarded by the Linux Foundation's Agentic AI Foundation. Key takeaways:

**Architecture:**
- Client-server model with JSON-RPC 2.0
- stdio for local, Streamable HTTP for remote
- Three primitives: Tools, Resources, Prompts

**Building Servers:**
- Python: FastMCP with decorators
- TypeScript: McpServer with Zod schemas
- Focus on single-purpose, well-documented tools

**Security:**
- OAuth 2.1 for HTTP transports
- Permission scoping and input validation
- Output sanitization and audit logging
- Sandboxing for untrusted servers

**Production:**
- Container-based deployment with health checks
- Horizontal scaling with stateless design
- Prometheus metrics and structured logging
- Version management and graceful degradation

**Integration:**
- Claude Desktop configuration files
- Cursor IDE project-level configs
- Custom clients with multi-server orchestration
- Fallback and circuit breaker patterns

For the latest updates, refer to the [official MCP documentation](https://modelcontextprotocol.io) and the [AAIF GitHub organization](https://github.com/agenticaiorg).

---

## References

**Official Documentation:**
- [Model Context Protocol Documentation](https://modelcontextprotocol.io)
- [MCP Specification](https://modelcontextprotocol.io/specification)
- [Python SDK (mcp-python)](https://github.com/modelcontextprotocol/python-sdk)
- [TypeScript SDK (mcp)](https://github.com/modelcontextprotocol/typescript-sdk)

**Foundation and Governance:**
- [Linux Foundation AAIF Announcement](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)
- [Anthropic MCP Donation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- [MCP Joins AAIF Blog](http://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/)

**Security Resources:**
- [MCP Security Best Practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [OWASP MCP Security Guide](https://genai.owasp.org/resource/cheatsheet-a-practical-guide-for-securely-using-third-party-mcp-servers-1-0/)
- [Red Hat MCP Security Analysis](https://www.redhat.com/en/blog/model-context-protocol-mcp-understanding-security-risks-and-controls)

**Community:**
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [MCP Examples](https://modelcontextprotocol.io/examples)
