#!/usr/bin/env python3
"""
Test comprehensive tool suite including data access and web tools.
"""

from tools.tool_interface import ToolManager

def test_data_tools():
    """Test data access tools."""
    print("Testing data access tools...")
    tm = ToolManager()
    
    # Test system info
    print("\n1. Testing system info...")
    result = tm.execute_tool("system_info", include_disk_usage=True)
    print(f"System info: {result.success}")
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    
    # Test data indexing (small test)
    print("\n2. Testing data indexing...")
    result = tm.execute_tool("index_data", 
                           root_paths=["/Users/axisthornllc/Documents/qwen_training"],
                           max_depth=3)
    print(f"Data indexing: {result.success} - {result.output}")
    
    # Test data search
    print("\n3. Testing data search...")
    result = tm.execute_tool("search_data", query="training", file_type=".py", limit=5)
    print(f"Data search: {result.success}")
    if result.success:
        print(result.output[:300] + "..." if len(result.output) > 300 else result.output)

def test_web_tools():
    """Test web access tools."""
    print("\nTesting web tools...")
    tm = ToolManager()
    
    # Test web search
    print("\n1. Testing web search...")
    result = tm.execute_tool("web_search", query="Python machine learning", num_results=3)
    print(f"Web search: {result.success}")
    if result.success:
        print(result.output[:500] + "..." if len(result.output) > 500 else result.output)
    
    # Test web scraping
    print("\n2. Testing web scraping...")
    result = tm.execute_tool("web_scrape", url="https://httpbin.org/html", max_length=1000)
    print(f"Web scraping: {result.success}")
    if result.success:
        print(result.output[:300] + "..." if len(result.output) > 300 else result.output)
    
    # Test website monitoring
    print("\n3. Testing website monitoring...")
    result = tm.execute_tool("web_monitor", 
                           urls=["https://httpbin.org/status/200", "https://httpbin.org/status/404"],
                           check_content=True)
    print(f"Website monitoring: {result.success}")
    if result.success:
        print(result.output[:400] + "..." if len(result.output) > 400 else result.output)

def test_comprehensive_access():
    """Test comprehensive data access capabilities."""
    print("\nTesting comprehensive access...")
    tm = ToolManager()
    
    # Show all available tools
    tools = tm.get_available_tools()
    print(f"\nAvailable tools ({len(tools)}):")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")
    
    # Test bulk access to current directory
    print(f"\n4. Testing bulk access...")
    result = tm.execute_tool("bulk_access",
                           paths=["/Users/axisthornllc/Documents/qwen_training"],
                           include_patterns=["*.md", "*.py"],
                           recursive=False,
                           max_file_size=5000)
    print(f"Bulk access: {result.success}")
    if result.success and result.metadata:
        print(f"Accessed {result.metadata.get('files_accessed', 0)} files")

if __name__ == "__main__":
    test_data_tools()
    test_web_tools()
    test_comprehensive_access()
    print("\nComprehensive tool testing complete.")