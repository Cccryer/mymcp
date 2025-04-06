import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llmclient = OpenAI(
            # api_key="sk-5c2ba0b53ccb4fbb8388c9f7158b0232",
            # base_url="https://api.deepseek.com",
            base_url="https://api.agicto.cn/v1",
            api_key="sk-BG7BGtHCOFyrx7btnF3gh8ylBeUzVa1AgAewRzcYKYc6QPYS",
            timeout=10,
            max_retries=3
        )

    # 使用sse连接
    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""
        # Store the context managers so they stay alive
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        # Initialize
        await self.session.initialize()

        # List available tools to verify connection
        print("Initialized SSE client...")
        print("Listing tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    # 使用stdio连接
    async def connect_to_server(self, server_script_path: str):

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

    async def process_query(self, query: str) -> str:
        """Process a query using MCP and available tools"""
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

        # print("available_tools", json.dumps(available_tools, indent=2))
        print("call openai")
        response = self.llmclient.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            tools=available_tools,
            tool_choice="auto"
        )
        print("openai response", response.to_json())
        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for choice in response.choices:
            if choice.message.content:
                final_text.append(choice.message.content)
                assistant_message_content.append(choice.message.content)
            elif choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                print("result", result.model_dump_json())
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(choice.message.to_dict())
                messages.extend(assistant_message_content)
                print("--------------------------------")
                messages.append({
                    "role": "tool",
                    "content": result.content[0].text,
                    "tool_call_id": tool_call.id
                })
                print("messages", json.dumps(messages, indent=2))
                # Get next response from OpenAI
                response = self.llmclient.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                print("openai response", response.to_json())

                final_text.append(response.choices[0].message.content)

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
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())




