from typing import Dict, TypedDict, List, Annotated, Sequence, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
import json

load_dotenv()

def add_or_update(current: list, new: list) -> list:
    return new if new else current

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tools: Annotated[List[dict], add_or_update]
    current_tool_calls: Annotated[List[dict], add_or_update]
    tool_results: Annotated[List[dict], add_or_update]
    final_response: Optional[str]

class MCPToolManager:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.available_tools = []

    async def initialize(self, server_script_path: str):
        """Initialize MCP connection and cache tools"""
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
        return self.available_tools

    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        result = await self.session.call_tool(tool_name, tool_args)
        return str(result.content) if result.content is not None else ""

    async def cleanup(self):
        await self.exit_stack.aclose()

class LangGraphMCPWorkflow:
    def __init__(self, mcp_manager: MCPToolManager):
        self.mcp_manager = mcp_manager
        self.system_template = """You are Plansom bot, a professional AI assistant. Your role is to:
1. Analyze the query and determine if tools are needed
2. If tools are needed, use them and incorporate their results
3. Provide a clear, direct response that answers the specific question
4. Use natural language and maintain second-person perspective ("you"/"your")
5. Present information without meta-commentary
6. Provide specific numbers and metrics when relevant
7. Keep responses focused and concise

If tool results are provided, incorporate them naturally into your response.
Keep responses clear, direct, and focused on answering the query.

Important formatting rules:
- Never use phrases like "Here's a response" or explain what you're doing
- Never use "I" or "we" - always use "you" or "your"
- Present information in clear, natural language
- Keep essential information only
- Maintain consistent structure throughout"""

    async def process_query(self, state: State) -> Dict:
        """Single step to analyze query, execute tools if needed, and generate response"""
        try:
            # Debug logging
            print(f"\nProcessing query with tools: {[t['function']['name'] for t in state['tools']]}")
            
            # Start by executing any tools that were previously called
            tool_results = []
            tool_messages = []
            
            if state.get("current_tool_calls"):
                for tool_call in state["current_tool_calls"]:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    result = await self.mcp_manager.execute_tool(tool_name, tool_args)
                    tool_results.append({
                        'tool': tool_name,
                        'result': result
                    })
                    tool_messages.append(FunctionMessage(
                        name=tool_name,
                        content=result
                    ))

            # Construct messages for the single LLM call
            messages = [
                {
                    "role": "system",
                    "content": self.system_template
                }
            ]

            # If we have tool results, include them in the context
            query_content = state['messages'][-1].content
            if tool_results:
                tool_results_text = "\n".join(
                    f"Tool {r['tool']} returned: {r['result']}" 
                    for r in tool_results
                )
                query_content = f"{query_content}\n\nPrevious Tool Results:\n{tool_results_text}"
                
            messages.append({
                "role": "user",
                "content": query_content
            })

            print("\nSending request to LLM...")
            # Single LLM call that either uses tools or generates final response
            response = self.mcp_manager.groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                tools=state["tools"],
                temperature=0.3,
                max_tokens=1000
            )
            print("Received response from LLM")
            
            assistant_message = response.choices[0].message
            print(f"Tool calls present: {hasattr(assistant_message, 'tool_calls')}")
            
            # Process the assistant's response
            content = assistant_message.content or "I need more information to provide a response."
            
            # If the response includes tool calls
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                print(f"\nTool calls requested: {[t.function.name for t in assistant_message.tool_calls]}")
                return {
                    "current_tool_calls": assistant_message.tool_calls,
                    "tool_results": [],  # Reset tool results for next iteration
                    "messages": [AIMessage(content=content)]
                }
            
            # Final response with any tool results
            final_content = content
            if tool_results:
                final_content = f"{content}\n\nIncorporating tool results from: {', '.join(r['tool'] for r in tool_results)}"
            
            return {
                "final_response": final_content,
                "tool_results": tool_results,
                "messages": [AIMessage(content=final_content)]
            }
            
        except Exception as e:
            print(f"\nError in process_query: {str(e)}")
            return {
                "final_response": f"Error processing query: {str(e)}",
                "messages": [AIMessage(content=f"Error processing query: {str(e)}")]
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return {
                "final_response": error_msg,
                "messages": [AIMessage(content=error_msg)]
            }

    def create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(State)
        
        # Add processing node
        workflow.add_node("process_query", self.process_query)
        
        # Define routing logic
        def router(state: State) -> str:
            if state.get("final_response"):
                return "end"
            if state.get("current_tool_calls"):
                return "process_query"  # Loop back to process tools
            return "end"
        
        # Add edges with routing
        workflow.add_conditional_edges(
            "process_query",
            router,
            {
                "process_query": "process_query",
                "end": END
            }
        )
        
        workflow.set_entry_point("process_query")
        
        return workflow

async def main(server_script_path: str):
    mcp_manager = MCPToolManager()
    
    try:
        # Initialize MCP connection and tools
        tools = await mcp_manager.initialize(server_script_path)
        
        # Create workflow instance
        workflow = LangGraphMCPWorkflow(mcp_manager)
        graph = workflow.create_graph().compile()
        
        print("\nLangGraph MCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
                
            # Initialize state
            state: State = {
                "messages": [HumanMessage(content=query)],
                "tools": tools,
                "current_tool_calls": [],
                "tool_results": [],
                "final_response": None
            }
            
            try:
                final_state = await graph.ainvoke(state)
                response = final_state.get("final_response")
                if response:
                    print("\n" + response)
                else:
                    print("\nNo response generated. Please try again.")
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
    finally:
        await mcp_manager.cleanup()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python langgraph_client.py <path_to_server_script>")
        sys.exit(1)
    
    asyncio.run(main(sys.argv[1]))