from typing import Any
import asyncio
import httpx
import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import asyncpg
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import sys

load_dotenv(override=True)

# Define allowed tables and operations
ALLOWED_TABLES = {
    "goals": ["SELECT"],
    "tasks": ["SELECT"],
   
   
}

GOAL_OWNER_ID =576

DB_CONFIG = {
    "host": os.getenv('DB_HOST', 'localhost'),
    "database": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "port": int(os.getenv('DB_PORT', '5432')),
    "ssl": os.getenv('DB_SSL', 'require'),

}

def validate_config():
    missing_vars = []
    for key in ['database', 'user', 'password']:
        if not DB_CONFIG.get(key):
            missing_vars.append(key)
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )

class DatabaseConnection:
    def __init__(self):
        self.pool = None
        self._pool_config = {
            **DB_CONFIG,
            'min_size': 2,          # Minimum number of connections in pool
            'max_size': 10,         # Maximum number of connections in pool
            'command_timeout': 60,   # Command timeout in seconds
            'max_inactive_connection_lifetime': 300.0,
            'max_queries': 50000
           
        }
    
    async def connect(self):
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(**self._pool_config)
                if not self.pool:
                    raise ConnectionError("Failed to create connection pool")
            except Exception as e:
                error_msg = f"Failed to connect to database: {str(e)}"
                if isinstance(e, asyncpg.exceptions.TooManyConnectionsError):
                    error_msg += "\nToo many database connections. Try reducing pool size or closing unused connections."
                raise ConnectionError(error_msg)
    
    async def close(self):
        if self.pool:
            await self.pool.close()
    
    async def execute_query(self, query: str) -> list[dict]:
        if not self.pool:
            await self.connect()
        
        try:
            async with self.pool.acquire() as connection:
                # Set statement timeout for safety
                await connection.execute('SET statement_timeout = 30000')  
                records = await connection.fetch(query)
                return [dict(record) for record in records]
                
        except asyncpg.PostgresError as e:
            error_msg = f"Database query error: {str(e)}"
            if isinstance(e, asyncpg.exceptions.QueryCanceledError):
                error_msg += "\nQuery timed out. Try optimizing your query or reducing the data size."
            raise ValueError(error_msg)

# Create server and database instances
server = Server("agents")
db = DatabaseConnection()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools with their schemas."""
    return [
        types.Tool(
            name="run-query",
            description = f"""You are the POSTGRES AGENT, an advanced AI database retriever for PLANSOM, designed to execute PostgreSQL queries on the tables: goals and tasks. Your expertise lies in retrieving precise and relevant information from a PostgreSQL database based on the questions you receive. Your primary role is to understand the provided question, execute the necessary queries, and deliver accurate and professional responses by fetching the required data from the designated tables.
                    Execute a PostgreSQL query on the tables {ALLOWED_TABLES}, leveraging their relationships to fetch accurate data.

                    ### Key Context:
                    PLANSOM is a productivity tracking tool that enables users to:
                    - Create and manage goals, subgoals, and tasks.
                    - Track progress and collaborate effectively within teams and organizations.
                    - Maintain a hierarchical structure:
                    - **Goal** → **Subgoal** → **Task**

                    Description of Tables goals and tasks:
                    the `goal_id` column in the `tasks` table is a foreign key that references the `id` column in the `goals` table. .
                    The relationship between the two tables can be represented as:`goals` (one) → `tasks` (many).
                    This is a one-to-many relationship, where one goal can have multiple tasks, but each task is associated with only one goal.
                    

                  ### Database Schema Overview:
                    1. **Goals Table**:
                    - Stores details about goals and subgoals.
                    - Primary Columns:

                    1. `id`: a unique identifier for the goal
                    2. `goal_creator_id`: the ID of the user who created the goal
                    3. `goal_owner_id`: the ID of the user who owns the goal
                    4. `goal_parent_id`: the ID of the parent goal (if applicable)
                    5. `organization_id`: the ID of the organization to which the goal belongs
                    6. `team_id`: the ID of the team to which the goal belongs
                    7. `order`: the order or priority of the goal
                    8. `deadline`: the deadline for the goal                       
                    9. `created_at`: the timestamp when the goal was created
                    10. `updated_at`: the timestamp when the goal was last updated
                    11. `goal_accepted`: a boolean indicating whether the goal has been accepted
                    12. `total_time_aligned`: the total time aligned with the goal
                    13. `total_completed_time`: the total time completed towards the goal
                    14. `goal_archived`: a boolean indicating whether the goal has been archived
                    15. `is_superparent`: a boolean indicating whether the goal is a superparent
                    16. `name`: the name of the goal
                    17. `description`: a brief description of the goal
                    18. `goal_type`: the type of goal (e.g. objective, key result, etc.)
                    19. `status`: status of the goal
                    20. `on_time_status`: the on-time status of the goal
                    21. `goal_success`: the success rate of the goal

                    2. **Tasks Table**:
                    The tasks table stores all data related to tasks, with the following columns:
                    1. `id` (primary key, bigint): a unique identifier for each task
                    2. `name` (varchar(255)): the name of the task
                    3. `description` (text): a brief description of the task
                    4. `task_type` (varchar(100)): the type of task (e.g., "development", "design", etc.)
                    5. `task_status` (varchar(100)): the current status of the task (e.g., "open", "in progress", "closed","start","pause","stop","resume","done","scheduled","to_be_scheduled","to_approved","late","completed","not completed" etc.)
                    6. `task_accepted` (boolean): whether the task has been accepted or not
                    7. `task_approval` (boolean): whether the task has been approved or not
                    8. `task_impact` (varchar(100)): the impact of the task (e.g., "high", "medium", "low", etc.)
                    9. `task_control` (varchar(100)): the control of the task (e.g., "automated", "manual", etc.)
                    10. `risk_status` (varchar(100)): the risk status of the task (e.g., "high", "medium", "low", etc.)
                    11. `task_effort` (double precision): the estimated effort required to complete the task
                    12. `task_schedule` (timestamp with time zone): the scheduled start date and time of the task
                    13. `task_schedule_end` (timestamp with time zone): the scheduled end date and time of the task
                    14. `on_time_status` (varchar(50)): the on-time status of the task (e.g., "on time", "On time", "late","Late","ok","Ok" etc.)
                    15. `deadline_status` (varchar(50)): the deadline status of the task (e.g., "met", "missed", etc.)  
                    16. `amount_late` (double precision): the amount of time the task is late   
                    17. `task_success` (varchar(50)): the success status of the task (e.g., "hit", "beat", "Miss","not completed","completed".)
                    18. `task_pulse` (varchar(50)): the pulse status of the task (e.g., "healthy", "unhealthy", etc.)
                    19. `created_at` (timestamp with time zone): the date and time the task was created
                    20. `updated_at` (timestamp with time zone): the date and time the task was last updated

                    21. `task_time_logged` (double precision): the total time logged for the task
                    22. `task_timestamp` (timestamp with time zone): the timestamp of the task
                    23. `task_value` (double precision): the value of the task
                    24. `task_time_slot` (jsonb): the time slot of the task
                    25. `task_completed` (boolean): whether the task is completed or not
                    26. `goal_time_logged` (boolean): whether the goal time is logged or not
                    27. `task_completed_time` (timestamp with time zone):this field is for to check date and time the task was completed
                    28. `aligned_time_to_goal` (boolean): whether the task is aligned with the goal time or not
                    29. `goal_id` (bigint): the ID of the goal associated with the task
                    30. `organization_id` (bigint): the ID of the organization associated with the task
                    31. `task_creator_id` (bigint): the ID of the user who created the task
                    32. `task_owner_id` (bigint): the ID of the user who owns the task
                    33. `task_parent_id` (bigint): the ID of the parent task
                    34. `team_id` (bigint): the ID of the team associated with the task
                    35. `task_order` (integer): the order of the task
                    36. `status` (varchar(100)): the status of the task  (eg. "Active", "active","Inactive", "inactive"),

                    How to Approach a Question:
                    
                    You will only retrive data for for goal_owner_id = {GOAL_OWNER_ID}
                    Relationships:
                    1. goals.goal_owner_id links goals to their owner.
                    2. goals.id = tasks.goal_id links tasks to their parent goal.

                    Key Fetching Logic:
                    - Fetch *goals* by filtering goal_owner_id = {GOAL_OWNER_ID}.
                    - to Fetch *tasks* associated with those goals and all data in tasks tableusing `tasks.goal_id = goals.id`and  also refer following query for help.
                    -SELECT g.id AS goal_id,
                            g.name AS goal_name,
                            g.description AS goal_description,
                            t.id AS task_id,
                            t.name AS task_name,quit
                            t.description AS task_description,
                            t.task_status,
                            t.deadline_time
                            t.created_at
                        FROM 
                            goals g
                        LEFT JOIN 
                            tasks t ON g.id = t.goal_id
                        WHERE 
                            g.goal_owner_id ={GOAL_OWNER_ID};

                    You will only answer the question recived as precisely as possible, do not give extra and unwanted information.
                    Break down the question into its components related to goals and tasks.
                    Execute the appropriate PostgreSQL queries to fetch the necessary data.
                    Respond professionally by rephrasing the answer clearly, politely, and accurately.
                    """,



            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute (SELECT queries or other related queries)",
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
    """Handle tool execution requests."""
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
               
                return [types.TextContent(
                    type="text",
                    text=str(results)
                )]
            else:
               
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
    # Validate configuration before starting
    validate_config()
    
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
    except Exception as e:
        print(f"Server error: {str(e)}", file=sys.stderr)
        raise
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())



 