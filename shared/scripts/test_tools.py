#!/usr/bin/env python3
"""
Test script for tool functionality.
"""

from tools.tool_interface import ToolManager, execute_tool_calls

def test_tool_interface():
    """Test the tool interface system."""
    print("Testing tool interface...")
    
    # Initialize tool manager
    tm = ToolManager()
    
    # Test file operations
    print("\n1. Testing file operations...")
    result = tm.execute_tool("write_file", file_path="test_output.txt", content="Test content")
    print(f"Write file: {result.success} - {result.output}")
    
    result = tm.execute_tool("read_file", file_path="test_output.txt")
    print(f"Read file: {result.success} - {result.output[:50]}...")
    
    # Test bash command
    print("\n2. Testing bash command...")
    result = tm.execute_tool("bash", command="echo 'Hello from bash'")
    print(f"Bash: {result.success} - {result.output.strip()}")
    
    # Test Python execution
    print("\n3. Testing Python execution...")
    result = tm.execute_tool("python_exec", code="print('Hello from Python'); print(2 + 2)")
    print(f"Python: {result.success} - {result.output.strip()}")
    
    # Test HTTP request
    print("\n4. Testing HTTP request...")
    result = tm.execute_tool("http_request", url="https://httpbin.org/get")
    print(f"HTTP: {result.success} - Status: {result.metadata.get('status_code') if result.metadata else 'N/A'}")
    
    # Test tool call parsing
    print("\n5. Testing tool call parsing...")
    test_response = """Here is the result:
TOOL_CALL: {"tool": "bash", "parameters": {"command": "date"}}
The command will show the current date."""
    
    final_response = execute_tool_calls(test_response, tm)
    print(f"Tool call execution result:\n{final_response}")

def test_available_tools():
    """Test listing available tools."""
    tm = ToolManager()
    tools = tm.get_available_tools()
    
    print(f"\nAvailable tools ({len(tools)}):")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    print("\nTool format for prompt:")
    print(tm.format_tools_for_prompt()[:500] + "...")

if __name__ == "__main__":
    test_tool_interface()
    test_available_tools()
    print("\nTool testing complete.")