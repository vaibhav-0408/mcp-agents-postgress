from typing import Any
import asyncio
import httpx
from datetime import datetime, timezone, timedelta
import asyncpg
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

ALLOWED_TABLES = {
    "goals": ["SELECT"],
   
    "tasks": ["SELECT"],
  
}

DB_CONFIG = {
    "host": "plansom.postgres.database.azure.com",
    "database": "postgres",
    "user": "plansomstaging",
    "password": "ortaOJKI8KIPaDn",
    "port": 5432
}
server = Server("agents")

class DatabaseConnection:
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(**DB_CONFIG)
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {str(e)}")
    
    async def close(self):
        if self.pool:
            await self.pool.close()
    
    async def execute_query(self, query: str) -> list[dict]:
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as connection:
             
                records = await connection.fetch(query)
                
              
                results = [dict(record) for record in records]
                return results
                
        except asyncpg.PostgresError as e:
            raise ValueError(f"Database query error: {str(e)}")

db = DatabaseConnection()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
      
       
       
        types.Tool(
            name="run-query",
            description=f"Execute a PostgreSQL query on table goals only ",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (SELECT queries only)",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format",
                        "enum": ["table", "json"],
                        "default": "table"
                    }
                },
                "required": ["query"],
            },
        ),
    ]







@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can fetch weather data and notify clients of changes.
    """
    if not arguments:
        raise ValueError("Missing arguments")

   

   

   
    if name == "run-query":
        query = arguments.get("query", "").strip()
        output_format = arguments.get("format", "table")

    
        if not query.lower().startswith("select"):
            return [types.TextContent(
                type="text",
                text="Error: Only SELECT queries are allowed for security reasons."
            )]

        try:
            # Execute query
            results = await db.execute_query(query)

            if not results:
                return [types.TextContent(
                    type="text",
                    text="Query executed successfully but returned no results."
                )]

            if output_format == "json":
                # Return results in JSON format
                return [types.TextContent(
                    type="text",
                    text=str(results)
                )]
            else:
                # Format results as a table
                headers = results[0].keys()
                header_row = " | ".join(str(h) for h in headers)
                separator = "-" * len(header_row)
                rows = [
                    " | ".join(str(row[h]) for h in headers)
                    for row in results
                ]
                
                table = f"{header_row}\n{separator}\n" + "\n".join(rows)
                return [types.TextContent(
                    type="text",
                    text=table
                )]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error executing query: {str(e)}"
            )]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    try:
        # Initialize database connection
        await db.connect()
        
        # Run the server using stdin/stdout streams
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="agents",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        # Clean up database connection
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())







