from typing import Dict, TypedDict, List, Annotated, Sequence, Union, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, FunctionMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
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

def delete_old_messages(state: State) -> Dict:
   messages = state["messages"]
   if len(messages) > 10:
       kept = messages[-10:]
       return {"messages": [RemoveMessage(id=m.id) for m in messages[:-10]]}
   return {}

class MCPToolManager:
   def __init__(self):
       self.session: Optional[ClientSession] = None
       self.exit_stack = AsyncExitStack()
       self.groq = Groq(api_key=os.getenv('GROQ_API_KEY'))
       self.available_tools = []
       
   async def initialize(self, server_script_path: str):
       is_python = server_script_path.endswith('.py')
       is_js = server_script_path.endswith('.js')
       if not (is_python or is_js):
           raise ValueError("Server script must be a .py or .js file")
           
       command = "python" if is_python else "node"
       server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
       
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
       self.system_template = """ou are Plansom bot, a professional AI assistant. Your role is to:
1. Analyze the query and determine needed information
2. Process results and provide clear, consistent insights
3. Focus only on answering the specific question
4. Use natural language and maintain second-person perspective ("you"/"your")
5. Present information directly without meta-commentary
6. Provide specific numbers and metrics when relevant
7. Keep responses focused and concise

Important formatting rules:
- Never use phrases like "Here's a response" or explain what you're doing
- Never use "I" or "we" - always use "you" or "your"
- Present information in clear, natural language
- Keep essential information only
- Maintain consistent structure throughout
Based on the query and tool results, provide a direct, clear response that:
1. Answers the specific question asked
2. Includes relevant metrics and insights
3. Uses natural language without meta-commentary
4. Maintains second-person perspective
5. Keeps focus on essential information
6. Maintain consistency in structure
"""
       self.memory = MemorySaver()

   async def analyze_query(self, state: State) -> Dict:
       current_query = state['messages'][-1].content
       memory_context = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" 
                               for m in state['messages'][:-1])
       
       system_content = f"{self.system_template}\n\nMemory Context:\n{memory_context}\n\nCurrent Query: {current_query}"
       
       try:
           response = self.mcp_manager.groq.chat.completions.create(
               model="llama3-70b-8192", 
               messages=[
                   {"role": "system", "content": system_content},
                   {"role": "user", "content": current_query}
               ],
               tools=state["tools"],
               temperature=0.3,
               max_tokens=1000
           )
           
           assistant_message = response.choices[0].message
           if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
               tool_results = []
               for tool_call in assistant_message.tool_calls:
                   try:
                       tool_name = tool_call.function.name
                       tool_args = json.loads(tool_call.function.arguments)
                       result = await self.mcp_manager.execute_tool(tool_name, tool_args)
                       tool_results.append({'tool': tool_name, 'result': result})
                   except Exception as e:
                       print(f"Tool execution error: {str(e)}")
               
               tool_results_text = "\n".join(f"{r['tool']}: {r['result']}" for r in tool_results)
               response = self.mcp_manager.groq.chat.completions.create(
                   model="llama3-70b-8192",
                   messages=[
                       {"role": "system", "content": system_content},
                       {"role": "user", "content": f"{current_query}\n\nResults:\n{tool_results_text}"}
                   ],
                   temperature=0.3,
                   max_tokens=1000
               )
               content = response.choices[0].message.content
               return {
                   "final_response": content,
                   "messages": state["messages"] + [AIMessage(content=content)]
               }
           
           return {
               "final_response": assistant_message.content,
               "messages": state["messages"] + [AIMessage(content=assistant_message.content)]
           }
       except Exception as e:
           return {
               "final_response": f"Error: {str(e)}",
               "messages": state["messages"]
           }

   def create_graph(self) -> StateGraph:
       workflow = StateGraph(State)
       workflow.add_node("analyze_query", self.analyze_query)
       workflow.add_node("delete_messages", delete_old_messages)
       workflow.add_edge("analyze_query", "delete_messages") 
       workflow.add_edge("delete_messages", END)
       workflow.set_entry_point("analyze_query")
       return workflow

async def main(server_script_path: str):
   mcp_manager = MCPToolManager()
   try:
       tools = await mcp_manager.initialize(server_script_path)
       workflow = LangGraphMCPWorkflow(mcp_manager)
       graph = workflow.create_graph().compile(checkpointer=workflow.memory)
       
       print("\nLangGraph MCP Client Started!")
       print("Type your queries or 'quit' to exit.")
       
       config = {"configurable": {"thread_id": "1"}}
       messages = []
       
       while True:
           query = input("\nQuery: ").strip()
           if query.lower() == 'quit':
               break

           messages.append(HumanMessage(content=query))
           state = {
               "messages": messages,
               "tools": tools,
               "current_tool_calls": [],
               "tool_results": [],
               "final_response": None
           }
           
           try:
               final_state = await graph.ainvoke(state, config)
               messages = final_state.get("messages", messages)
               
               print(f"\nMessages in memory: {len(messages)}")
               print("\n=== MEMORY STATE ===")
               print(f"Total Messages: {len(messages)}")
               print("\nMessages:")
               for i, msg in enumerate(messages):
                   print(f"{i}. {type(msg).__name__}: {msg.content}")
               print("==================\n")
               
               if final_state.get("final_response"):
                   print("\nResponse:", final_state["final_response"]) 
               
           except Exception as e:
               print(f"\nError: {str(e)}")
           
   finally:
       await mcp_manager.cleanup()

if __name__ == "__main__":
   import sys
   if len(sys.argv) != 2:
       print("Usage: python script.py <server_script>")
       sys.exit(1)
   
   asyncio.run(main(sys.argv[1]))