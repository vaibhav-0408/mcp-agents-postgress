import asyncio
import json
import os
import sys
import signal
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class OptimizedMCPClient:
    """An optimized Message Control Protocol client with Groq LLM integration"""
    
    def __init__(self):
        """Initialize the MCP client with required components"""
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.groq: Groq = Groq(api_key=self._get_api_key())
        self.available_tools: List[Dict[str, Any]] = []
        self.stdio: Any = None
        self.write: Any = None
        
        # System prompt template for LLM interactions
        self.system_prompt = """You are Plansom bot, a professional AI assistant capable of both using tools for specific queries and engaging in general conversation. Your role is to:
1. First analyze if the query requires using available tools or if it's a general conversation query
2. For tool-based queries:
    
    - Analyze the query and determine needed information
    - Process results and provide clear, consistent insights
    - Focus only on answering the specific question
    - Use natural language and maintain second-person perspective ("you"/"your")
    - Present information directly without meta-commentary
    - Provide specific numbers and metrics when relevant
    - Keep responses focused and concise
3. For general conversation:
   - Provide helpful, informative responses
   - Draw from your general knowledge
   - Maintain a natural, conversational tone
   - Be honest about limitations

Important formatting rules:
- Never use phrases like "Here's a response" or explain what you're doing
- Never use "I" or "we" - always use "you" or "your"
- Present information in clear, natural language
- Keep essential information only
- Maintain consistent structure throughout

Current query: {query}
Based on the query and tool results, provide a direct, clear response that:
1. Answers the specific question asked
2. Includes relevant metrics and insights
3. Uses natural language without meta-commentary
4. Maintains second-person perspective
5. Keeps focus on essential information
6. Maintain consistency in structure

Tool Results:
{tool_results}"""

    @staticmethod
    def _get_api_key() -> str:
        """Get Groq API key from environment"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        return api_key

    async def connect_to_server(self, server_script_path: str) -> bool:
        """Connect to an MCP server and cache available tools"""
        try:
            if not os.path.exists(server_script_path):
                raise FileNotFoundError(f"Server script not found: {server_script_path}")
                
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")
            
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=os.environ.copy()
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            
            response = await self.session.list_tools()
            if not response.tools:
                raise ValueError("No tools available from server")
                
            self.available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]
            
            print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.available_tools])
            return True
            
        except Exception as e:
            print(f"Failed to connect to server: {str(e)}")
            await self.cleanup()
            return False

    async def process_query(self, query: str) -> str:
        """Process a query using Groq with tool integration"""
        if not self.session:
            return "Not connected to server. Please connect first."
            
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt.format(query=query if query else "", tool_results="No tool results yet")
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            response = self.groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                tools=self.available_tools,
                temperature=0.3,
                max_tokens=1000,
                timeout=30
            )

            assistant_message = response.choices[0].message
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    try:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        result = await asyncio.wait_for(
                            self.session.call_tool(tool_name, tool_args),
                            timeout=10
                        )
                        tool_results.append(str(result.content) if result.content is not None else "")
                    except asyncio.TimeoutError:
                        tool_results.append("Tool execution timed out")
                    except Exception as e:
                        tool_results.append(f"Tool execution failed: {str(e)}")
                
                try:
                    return assistant_message.content.format(*tool_results)
                except (KeyError, IndexError, ValueError) as e:
                    return f"Error formatting response with tool results: {str(e)}"

            if assistant_message and hasattr(assistant_message, 'content'):
                return str(assistant_message.content) if assistant_message.content is not None else "No content in response"
            return "No response generated"

        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        if not self.session:
            print("Error: Not connected to server")
            return
            
        print("\nOptimized MCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        try:
            while True:
                try:
                    query = input("\nQuery: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                        
                    response = await self.process_query(query)
                    if response is not None:
                        print("\n" + str(response))
                    else:
                        print("\nNo response received")
                    
                except KeyboardInterrupt:
                    print("\nReceived interrupt signal...")
                    break
                except EOFError:
                    print("\nInput stream closed...")
                    break
                except Exception as e:
                    print(f"\nError processing query: {str(e)}")
                    
        finally:
            print("\nShutting down client...")
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources and close connections"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            if self.stdio:
                self.stdio = None
            if self.write:
                self.write = None
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def main():
    """Main entry point with signal handling"""
    if len(sys.argv) != 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    client = None
    
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        if client:
            asyncio.create_task(client.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    async def run():
        nonlocal client
        client = OptimizedMCPClient()
        try:
            if await client.connect_to_server(sys.argv[1]):
                await client.chat_loop()
        finally:
            await client.cleanup()
    
    asyncio.run(run())

if __name__ == "__main__":
    main()