import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  

class MCPClient:
    def __init__(self):
       
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
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
        
       
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Groq and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

       
        response = self.groq.chat.completions.create(
            model="llama3-70b-8192", 
            messages=messages,
            tools=available_tools,
            temperature=0.7,
            max_tokens=1000
        )

        tool_results = []
        final_text = []

        assistant_message = response.choices[0].message
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                
                tool_args = {}
                if hasattr(tool_call.function, 'arguments'):
                 
                    if isinstance(tool_call.function.arguments, str):
                        try:
                            import json
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing tool arguments: {e}")
                            continue
                    else:
                        tool_args = tool_call.function.arguments
                elif hasattr(tool_call, 'parameters'):
                  
                    tool_args = tool_call.parameters
                
            
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                # final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

               
                tool_result_content = str(result.content) if result.content is not None else ""

               
                messages.extend([
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_content
                    }
                ])

                response = self.groq.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                final_text.append(response.choices[0].message.content)
        else:
            final_text.append(assistant_message.content)

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

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__== "__main__":
    import sys
    asyncio.run(main())