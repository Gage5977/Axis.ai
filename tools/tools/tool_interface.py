#!/usr/bin/env python3
"""
Tool interface system for objective Qwen model.
Provides access to external tools and functions.
"""

import json
import subprocess
import os
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseTool(ABC):
    """Base class for all tools."""
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

class FileReadTool(BaseTool):
    def get_name(self) -> str:
        return "read_file"
    
    def get_description(self) -> str:
        return "Read contents of a text file"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }
    
    def execute(self, file_path: str) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(False, "", f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ToolResult(True, content, metadata={"file_size": len(content)})
        except Exception as e:
            return ToolResult(False, "", str(e))

class FileWriteTool(BaseTool):
    def get_name(self) -> str:
        return "write_file"
    
    def get_description(self) -> str:
        return "Write content to a text file"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["file_path", "content"]
        }
    
    def execute(self, file_path: str, content: str) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(True, f"File written: {file_path}", 
                            metadata={"bytes_written": len(content.encode('utf-8'))})
        except Exception as e:
            return ToolResult(False, "", str(e))

class BashTool(BaseTool):
    def get_name(self) -> str:
        return "bash"
    
    def get_description(self) -> str:
        return "Execute bash commands"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                }
            },
            "required": ["command"]
        }
    
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                metadata={"return_code": result.returncode}
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout} seconds")
        except Exception as e:
            return ToolResult(False, "", str(e))

class HttpRequestTool(BaseTool):
    def get_name(self) -> str:
        return "http_request"
    
    def get_description(self) -> str:
        return "Make HTTP requests"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to request"
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, etc.)",
                    "default": "GET"
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers",
                    "default": {}
                },
                "data": {
                    "type": "string",
                    "description": "Request body data"
                }
            },
            "required": ["url"]
        }
    
    def execute(self, url: str, method: str = "GET", headers: Dict = None, data: str = None) -> ToolResult:
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or {},
                data=data,
                timeout=30
            )
            
            return ToolResult(
                success=200 <= response.status_code < 300,
                output=response.text,
                metadata={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "url": response.url
                }
            )
        except Exception as e:
            return ToolResult(False, "", str(e))

class PythonTool(BaseTool):
    def get_name(self) -> str:
        return "python_exec"
    
    def get_description(self) -> str:
        return "Execute Python code safely"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    
    def execute(self, code: str) -> ToolResult:
        try:
            # Create restricted environment
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round
                },
                'math': __import__('math'),
                'json': __import__('json'),
                'datetime': __import__('datetime')
            }
            
            # Capture output
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                exec(code, restricted_globals)
                output = sys.stdout.getvalue()
                return ToolResult(True, output)
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return ToolResult(False, "", str(e))

class ToolManager:
    """Manages available tools and executes tool calls."""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        default_tools = [
            FileReadTool(),
            FileWriteTool(),
            BashTool(),
            HttpRequestTool(),
            PythonTool()
        ]
        
        # Add data access tools
        try:
            from .data_access import (
                DataIndexTool, DataSearchTool, BulkDataAccessTool,
                ContentSearchTool, SystemInfoTool
            )
            default_tools.extend([
                DataIndexTool(),
                DataSearchTool(), 
                BulkDataAccessTool(),
                ContentSearchTool(),
                SystemInfoTool()
            ])
        except ImportError:
            pass
            
        # Add web tools
        try:
            from .web_tools import (
                WebSearchTool, WebScrapeTool, NewsSearchTool, WebMonitorTool
            )
            default_tools.extend([
                WebSearchTool(),
                WebScrapeTool(),
                NewsSearchTool(),
                WebMonitorTool()
            ])
        except ImportError:
            pass
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.get_name()] = tool
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their schemas."""
        return [
            {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "parameters": tool.get_parameters()
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            return ToolResult(False, "", f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        return tool.execute(**kwargs)
    
    def format_tools_for_prompt(self) -> str:
        """Format tools for inclusion in model prompt."""
        tools_info = []
        for tool in self.tools.values():
            tools_info.append(f"""
Tool: {tool.get_name()}
Description: {tool.get_description()}
Parameters: {json.dumps(tool.get_parameters(), indent=2)}
""")
        
        return f"""
Available Tools:
{''.join(tools_info)}

To use a tool, respond with:
TOOL_CALL: {{"tool": "tool_name", "parameters": {{"param1": "value1"}}}}
"""

def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Parse tool calls from model response."""
    tool_calls = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('TOOL_CALL:'):
            try:
                tool_call_json = line[10:].strip()  # Remove 'TOOL_CALL:'
                tool_call = json.loads(tool_call_json)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
    
    return tool_calls

def execute_tool_calls(response: str, tool_manager: ToolManager) -> str:
    """Execute tool calls found in response and return updated response."""
    tool_calls = parse_tool_calls(response)
    
    if not tool_calls:
        return response
    
    # Execute each tool call and append results
    updated_response = response
    
    for tool_call in tool_calls:
        tool_name = tool_call.get('tool')
        parameters = tool_call.get('parameters', {})
        
        result = tool_manager.execute_tool(tool_name, **parameters)
        
        if result.success:
            updated_response += f"\n\nTool Result ({tool_name}):\n{result.output}"
        else:
            updated_response += f"\n\nTool Error ({tool_name}): {result.error}"
    
    return updated_response