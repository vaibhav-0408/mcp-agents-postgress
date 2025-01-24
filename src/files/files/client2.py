import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any

load_dotenv()  # load environment variables from .env

class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = "claude-3-5-sonnet-20241022"

class ToolResponse(BaseModel):
    name: str
    description: str
    input_schema: dict

class APIResponse(BaseModel):
    response: str
    tools_used: Optional[List[Dict[str, Any]]] = None

app = FastAPI()

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.last_tools_used = []  # Track tool usage
        self._server_script_path = None

    @property
    def server_script_path(self):
        return self._server_script_path

    @server_script_path.setter
    def server_script_path(self, path: str):
        self._server_script_path = path

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        self.server_script_path = server_script_path
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        return tools

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # Reset tools used for new query
        self.last_tools_used = []
        
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        tool_results = []
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Track tool usage
                self.last_tools_used.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": result.content if hasattr(result, 'content') else str(result)
                })
                
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# Global MCP client instance
mcp_client = MCPClient()

@app.on_event("startup")
async def startup_event():
    """Initialize MCP client on startup"""
    try:
        await mcp_client.connect_to_server("files/server.py")  # Update path as needed
    except Exception as e:
        print(f"Error initializing MCP client: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await mcp_client.cleanup()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server_path": mcp_client.server_script_path}

@app.get("/tools", response_model=List[ToolResponse])
async def list_tools():
    """List all available LLM tools"""
    try:
        response = await mcp_client.session.list_tools()
        tools = [
            ToolResponse(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema
            )
            for tool in response.tools
        ]
        return tools
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

@app.post("/query", response_model=APIResponse)
async def process_query(request: QueryRequest):
    """Process a query using the LLM and available tools"""
    try:
        response = await mcp_client.process_query(request.query)
        return APIResponse(
            response=response,
            tools_used=mcp_client.last_tools_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def start_server(port: int = 8000):
    """Start the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=port)

async def run_cli():
    """Run in CLI mode"""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    try:
        await mcp_client.connect_to_server(sys.argv[1])
        await mcp_client.chat_loop()
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run as API server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        start_server(port)
    else:
        # Run as CLI
        asyncio.run(run_cli())