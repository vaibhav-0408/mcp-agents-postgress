import asyncio
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
import os
import sys
from groq import Groq
from dotenv import load_dotenv
import json
from datetime import datetime


load_dotenv()

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.history: List[Dict] = []
        self.max_turns = max_turns
        self.tool_history: List[Dict] = []

    def add_interaction(self, query: str, response: str, tool_calls: List[Dict] = None):
        """Add a new interaction to memory"""
        self.history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'tool_calls': tool_calls or []
        })

        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context_window(self, n_turns: int = None) -> str:
        """Get formatted conversation history for context window"""
        turns = self.history[-n_turns:] if n_turns else self.history
        context = []
        
        for turn in turns:
            context.append(f"User: {turn['query']}")
            if turn['tool_calls']:
                for tool_call in turn['tool_calls']:
                    context.append(f"Tool ({tool_call['tool']}): {tool_call['result']}")
            context.append(f"Assistant: {turn['response']}\n")
            
        return "\n".join(context)

    def get_relevant_tool_history(self, query: str, threshold: float = 0.3) -> List[Dict]:
        """Get relevant tool calls from history based on query similarity"""
        relevant_calls = []
        for turn in self.history:
            if turn['tool_calls']:
                # Simple word-based similarity check
                query_words = set(query.lower().split())
                turn_words = set(turn['query'].lower().split())
                similarity = len(query_words.intersection(turn_words)) / len(query_words.union(turn_words))
                
                if similarity > threshold:
                    relevant_calls.extend(turn['tool_calls'])
        
        return relevant_calls[:5]  # Return top 5 relevant tool calls


class OptimizedMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.available_tools = []
        self.memory = ConversationMemory(max_turns=10)

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
        """Process a query using Groq and available tools with memory integration"""
        # Enhanced system prompt with conversation history
        conversation_context = self.memory.get_context_window(n_turns=5)  # Get last 5 turns
        relevant_tools = self.memory.get_relevant_tool_history(query)
        
        system_prompt = f"""You are Plansom bot, a professional AI assistant. Consider the following conversation history and context:

Previous Conversation:
{conversation_context}

Relevant Previous Tool Usage:
{json.dumps(relevant_tools, indent=2)}

Your role is to:
1. Analyze the query and determine needed information
2. Process results and provide clear, consistent insights
3. Focus only on answering the specific question
4. Use natural language and maintain second-person perspective ("you"/"your")
5. Use previous context when relevant to provide more informed responses
6. Present information directly without meta-commentary
7. Provide specific numbers and metrics when relevant
8. Keep responses focused and concise

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
                "content": system_prompt
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
        tool_results = []
        final_text=[]

        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            # Process tool calls and collect results
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
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                print(final_text)

            if not tool_results:
                return "I am not able to gave answer"

            # Second LLM call for final response with all tool results
            final_prompt = f"""Based on the conversation history, query, and tool results, provide a direct, clear response that:
1. Answers the specific question asked
2. Includes relevant metrics and insights
3. Uses natural language without meta-commentary
4. Maintains second-person perspective
5. Keeps focus on essential information
6. Maintains consistency in structure
7. References relevant previous context when appropriate

Previous Conversation:
{conversation_context}

Tool Results:
{json.dumps(tool_results, indent=2)}

Original Query:
{query}"""

            messages = [
                {
                    "role": "system",
                    "content": final_prompt
                }
            ]

            final_response = self.groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            ).choices[0].message.content

            # Store the interaction in memory
            self.memory.add_interaction(query, final_response, tool_results)
            return final_response
        else:
            # If no tools were called, store the direct response
            self.memory.add_interaction(query, assistant_message.content)
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