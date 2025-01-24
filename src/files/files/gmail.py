from typing import Any
import asyncio
from datetime import datetime
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
import email
from email.mime.text import MIMEText
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# If modifying scopes, delete the token.json file
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

server = Server("gmail_reader")

def get_gmail_service():
    """Get authenticated Gmail service."""
    creds = None
    # The token.json file stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no valid credentials, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def parse_email_date(date_str: str) -> datetime:
    """Parse email date string to datetime object."""
    return datetime.strptime(date_str, '%Y/%m/%d')

def get_email_body(message):
    """Extract email body from message payload."""
    if 'parts' in message['payload']:
        for part in message['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                return base64.urlsafe_b64decode(part['body']['data']).decode()
    elif 'body' in message['payload']:
        return base64.urlsafe_b64decode(message['payload']['body']['data']).decode()
    return "No text content available"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available Gmail tools."""
    return [
        types.Tool(
            name="read-gmail",
            description="Read Gmail messages for a specific date",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY/MM/DD format",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of emails to return",
                        "default": 10
                    }
                },
                "required": ["date"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle Gmail tool execution."""
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "read-gmail":
        try:
            date_str = arguments.get("date")
            max_results = arguments.get("max_results", 10)
            
            # Validate date format
            target_date = parse_email_date(date_str)
            date_query = f"after:{date_str} before:{target_date.strftime('%Y/%m/%d')}"
            
            # Get Gmail service
            service = get_gmail_service()
            
            # Search for messages on the specified date
            results = service.users().messages().list(
                userId='me',
                q=date_query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return [types.TextContent(
                    type="text",
                    text=f"No emails found for date: {date_str}"
                )]
            
            # Fetch full message details and format output
            email_details = []
            for message in messages:
                msg = service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()
                
                # Extract headers
                headers = msg['payload']['headers']
                subject = next(h['value'] for h in headers if h['name'].lower() == 'subject')
                sender = next(h['value'] for h in headers if h['name'].lower() == 'from')
                date = next(h['value'] for h in headers if h['name'].lower() == 'date')
                
                # Get message body
                body = get_email_body(msg)
                
                email_details.append(
                    f"Subject: {subject}\n"
                    f"From: {sender}\n"
                    f"Date: {date}\n"
                    f"Body:\n{body}\n"
                    f"{'='*50}\n"
                )
            
            return [types.TextContent(
                type="text",
                text=f"Emails for {date_str}:\n\n" + "\n".join(email_details)
            )]
            
        except ValueError as e:
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Failed to retrieve emails: {str(e)}"
            )]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gmail_reader",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())