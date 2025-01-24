import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import sys
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

class OptimizedMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.available_tools = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server and cache available tools"""
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
        
        # Cache available tools during initialization
        response = await self.session.list_tools()
        self.available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.available_tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Groq and available tools with optimized prompting"""
        # Combined system prompt that handles both analysis and formatting
        system_prompt = """You are Plansom bot, a professional AI assistant capable of both using tools for specific queries and engaging in general conversation. Your role is to:
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

Current query: {query}"""

        messages = [
            {
                "role": "system",
                "content": system_prompt.format(query=query)
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # First LLM call for tool selection and initial processing
        response = self.groq.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            tools=self.available_tools,
            temperature=0.3,
            max_tokens=1000
        )

        assistant_message = response.choices[0].message
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            # Process tool calls and collect results
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                try:
                    tool_args = (json.loads(tool_call.function.arguments) 
                               if isinstance(tool_call.function.arguments, str)
                               else tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_args = getattr(tool_call, 'parameters', {})
                
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({
                    'tool': tool_name,
                    'result': str(result.content) if result.content is not None else ""
                })
            if not tool_results:
                return "I am not able to gave answer"

            # Second LLM call for final response with all tool results
            final_prompt = """Based on the query and tool results, provide a direct, clear response that:
1. Answers the specific question asked
2. Includes relevant metrics and insights
3. Uses natural language without meta-commentary
4. Maintains second-person perspective
5. Keeps focus on essential information
6. Maintain consistency in structure

Tool Results:
{tool_results}

Original Query:
{query}"""

            messages = [
                {
                    "role": "system",
                    "content": final_prompt.format(
                        tool_results="\n".join(f"{r['tool']}: {r['result']}" for r in tool_results),
                        query=query
                    )
                }
            ]

            final_response = self.groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content

            return final_response
        else:
            # If no tools were called, return the direct response
            return assistant_message.content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nOptimized MCP Client Started!")
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    async def run():
        client = OptimizedMCPClient()
        try:
            await client.connect_to_server(sys.argv[1])
            await client.chat_loop()
        finally:
            await client.cleanup()
    
    asyncio.run(run())