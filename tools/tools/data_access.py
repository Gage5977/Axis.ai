#!/usr/bin/env python3
"""
Comprehensive data access tools for Qwen model.
Provides access to all local data with proper indexing and search.
"""

import os
import json
import sqlite3
import csv
import subprocess
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import fnmatch
import hashlib
from datetime import datetime
import mimetypes

from .tool_interface import BaseTool, ToolResult

class DataIndexTool(BaseTool):
    """Index all files in specified directories."""
    
    def get_name(self) -> str:
        return "index_data"
    
    def get_description(self) -> str:
        return "Create searchable index of all local files and directories"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "root_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Root directories to index",
                    "default": ["/Users/axisthornllc"]
                },
                "exclude_patterns": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Patterns to exclude",
                    "default": [".*", "__pycache__", "node_modules", "*.log"]
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth",
                    "default": 10
                }
            }
        }
    
    def execute(self, root_paths: List[str] = None, exclude_patterns: List[str] = None, max_depth: int = 10) -> ToolResult:
        if root_paths is None:
            root_paths = ["/Users/axisthornllc"]
        if exclude_patterns is None:
            exclude_patterns = [".*", "__pycache__", "node_modules", "*.log", "*.pyc"]
        
        try:
            index_db = "/Users/axisthornllc/Documents/qwen_training/data_index.db"
            conn = sqlite3.connect(index_db)
            
            # Create index table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_index (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE,
                    name TEXT,
                    size INTEGER,
                    modified TIMESTAMP,
                    type TEXT,
                    extension TEXT,
                    hash TEXT,
                    directory TEXT,
                    depth INTEGER,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            indexed_count = 0
            
            for root_path in root_paths:
                if not os.path.exists(root_path):
                    continue
                    
                for root, dirs, files in os.walk(root_path):
                    # Calculate depth
                    depth = root.replace(root_path, '').count(os.sep)
                    if depth > max_depth:
                        continue
                    
                    # Filter directories
                    dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)]
                    
                    for file in files:
                        # Skip excluded files
                        if any(fnmatch.fnmatch(file, pattern) for pattern in exclude_patterns):
                            continue
                        
                        file_path = os.path.join(root, file)
                        
                        try:
                            stat = os.stat(file_path)
                            file_size = stat.st_size
                            modified = datetime.fromtimestamp(stat.st_mtime)
                            
                            # Get file type
                            mime_type, _ = mimetypes.guess_type(file_path)
                            extension = os.path.splitext(file)[1].lower()
                            
                            # Calculate hash for small files
                            file_hash = None
                            if file_size < 10 * 1024 * 1024:  # 10MB limit
                                try:
                                    with open(file_path, 'rb') as f:
                                        file_hash = hashlib.md5(f.read()).hexdigest()
                                except:
                                    pass
                            
                            # Insert into database
                            conn.execute("""
                                INSERT OR REPLACE INTO file_index 
                                (path, name, size, modified, type, extension, hash, directory, depth)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (file_path, file, file_size, modified, mime_type, extension, file_hash, root, depth))
                            
                            indexed_count += 1
                            
                        except Exception as e:
                            continue
            
            conn.commit()
            conn.close()
            
            return ToolResult(True, f"Indexed {indexed_count} files in database: {index_db}")
            
        except Exception as e:
            return ToolResult(False, "", str(e))

class DataSearchTool(BaseTool):
    """Search indexed data by various criteria."""
    
    def get_name(self) -> str:
        return "search_data"
    
    def get_description(self) -> str:
        return "Search indexed files by name, type, content, or metadata"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (filename, path, or content)"
                },
                "file_type": {
                    "type": "string",
                    "description": "Filter by file extension (.py, .txt, .json, etc.)"
                },
                "directory": {
                    "type": "string",
                    "description": "Search within specific directory"
                },
                "size_min": {
                    "type": "integer",
                    "description": "Minimum file size in bytes"
                },
                "size_max": {
                    "type": "integer", 
                    "description": "Maximum file size in bytes"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 50
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, file_type: str = None, directory: str = None, 
               size_min: int = None, size_max: int = None, limit: int = 50) -> ToolResult:
        try:
            index_db = "/Users/axisthornllc/Documents/qwen_training/data_index.db"
            if not os.path.exists(index_db):
                return ToolResult(False, "", "Data index not found. Run index_data first.")
            
            conn = sqlite3.connect(index_db)
            
            # Build query
            sql = "SELECT path, name, size, modified, type FROM file_index WHERE 1=1"
            params = []
            
            # Text search in name or path
            if query:
                sql += " AND (name LIKE ? OR path LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            # File type filter
            if file_type:
                if not file_type.startswith('.'):
                    file_type = '.' + file_type
                sql += " AND extension = ?"
                params.append(file_type)
            
            # Directory filter
            if directory:
                sql += " AND directory LIKE ?"
                params.append(f"%{directory}%")
            
            # Size filters
            if size_min is not None:
                sql += " AND size >= ?"
                params.append(size_min)
            
            if size_max is not None:
                sql += " AND size <= ?"
                params.append(size_max)
            
            sql += " ORDER BY modified DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(sql, params)
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return ToolResult(True, "No files found matching criteria")
            
            # Format results
            output = f"Found {len(results)} files:\n\n"
            for path, name, size, modified, file_type in results:
                size_mb = size / (1024 * 1024) if size > 1024 * 1024 else size / 1024
                size_unit = "MB" if size > 1024 * 1024 else "KB"
                output += f"{name}\n"
                output += f"  Path: {path}\n"
                output += f"  Size: {size_mb:.1f} {size_unit}\n"
                output += f"  Modified: {modified}\n"
                output += f"  Type: {file_type or 'unknown'}\n\n"
            
            return ToolResult(True, output, metadata={"count": len(results)})
            
        except Exception as e:
            return ToolResult(False, "", str(e))

class BulkDataAccessTool(BaseTool):
    """Access multiple files or directories at once."""
    
    def get_name(self) -> str:
        return "bulk_access"
    
    def get_description(self) -> str:
        return "Access multiple files or entire directories with filtering"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file or directory paths"
                },
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"}, 
                    "description": "File patterns to include (*.py, *.txt, etc.)",
                    "default": ["*"]
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search directories recursively",
                    "default": True
                },
                "max_file_size": {
                    "type": "integer",
                    "description": "Maximum file size to read (bytes)",
                    "default": 1048576
                }
            },
            "required": ["paths"]
        }
    
    def execute(self, paths: List[str], include_patterns: List[str] = None, 
               recursive: bool = True, max_file_size: int = 1048576) -> ToolResult:
        if include_patterns is None:
            include_patterns = ["*"]
        
        try:
            results = {}
            total_files = 0
            
            for path in paths:
                if not os.path.exists(path):
                    results[path] = {"error": "Path not found"}
                    continue
                
                if os.path.isfile(path):
                    # Single file
                    try:
                        stat = os.stat(path)
                        if stat.st_size <= max_file_size:
                            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            results[path] = {"content": content, "size": stat.st_size}
                        else:
                            results[path] = {"error": f"File too large ({stat.st_size} bytes)"}
                        total_files += 1
                    except Exception as e:
                        results[path] = {"error": str(e)}
                
                elif os.path.isdir(path):
                    # Directory
                    dir_results = {}
                    
                    if recursive:
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                if any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns):
                                    file_path = os.path.join(root, file)
                                    try:
                                        stat = os.stat(file_path)
                                        if stat.st_size <= max_file_size:
                                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()
                                            rel_path = os.path.relpath(file_path, path)
                                            dir_results[rel_path] = {"content": content, "size": stat.st_size}
                                            total_files += 1
                                    except Exception as e:
                                        rel_path = os.path.relpath(file_path, path)
                                        dir_results[rel_path] = {"error": str(e)}
                    else:
                        # Non-recursive
                        for file in os.listdir(path):
                            if os.path.isfile(os.path.join(path, file)):
                                if any(fnmatch.fnmatch(file, pattern) for pattern in include_patterns):
                                    file_path = os.path.join(path, file)
                                    try:
                                        stat = os.stat(file_path)
                                        if stat.st_size <= max_file_size:
                                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()
                                            dir_results[file] = {"content": content, "size": stat.st_size}
                                            total_files += 1
                                    except Exception as e:
                                        dir_results[file] = {"error": str(e)}
                    
                    results[path] = dir_results
            
            # Format output
            output = f"Accessed {total_files} files from {len(paths)} paths\n\n"
            output += json.dumps(results, indent=2)
            
            return ToolResult(True, output, metadata={"files_accessed": total_files, "paths": len(paths)})
            
        except Exception as e:
            return ToolResult(False, "", str(e))

class ContentSearchTool(BaseTool):
    """Search file contents using grep-like functionality."""
    
    def get_name(self) -> str:
        return "search_content"
    
    def get_description(self) -> str:
        return "Search within file contents using text patterns or regex"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern or regex to search for"
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Directories or files to search in",
                    "default": ["/Users/axisthornllc"]
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to include",
                    "default": [".txt", ".py", ".js", ".json", ".md", ".yaml", ".yml", ".csv"]
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case sensitive search",
                    "default": False
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of context lines around matches",
                    "default": 2
                }
            },
            "required": ["pattern"]
        }
    
    def execute(self, pattern: str, paths: List[str] = None, file_types: List[str] = None,
               case_sensitive: bool = False, context_lines: int = 2) -> ToolResult:
        if paths is None:
            paths = ["/Users/axisthornllc"]
        if file_types is None:
            file_types = [".txt", ".py", ".js", ".json", ".md", ".yaml", ".yml", ".csv"]
        
        try:
            # Use ripgrep if available, otherwise use grep
            cmd = ["rg", "--no-heading", "--line-number"]
            
            if not case_sensitive:
                cmd.append("--ignore-case")
            
            cmd.extend(["--context", str(context_lines)])
            
            # Add file type filters
            for ext in file_types:
                cmd.extend(["--type-add", f"custom:{ext}"])
            cmd.extend(["--type", "custom"])
            
            cmd.append(pattern)
            cmd.extend(paths)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return ToolResult(True, result.stdout)
            elif result.returncode == 1:
                # No matches found
                return ToolResult(True, "No matches found")
            else:
                # Fallback to grep
                grep_cmd = ["grep", "-r", "-n"]
                if not case_sensitive:
                    grep_cmd.append("-i")
                grep_cmd.extend(["-A", str(context_lines), "-B", str(context_lines)])
                
                # Add file includes
                for ext in file_types:
                    grep_cmd.extend(["--include", f"*{ext}"])
                
                grep_cmd.append(pattern)
                grep_cmd.extend(paths)
                
                result = subprocess.run(grep_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    return ToolResult(True, result.stdout)
                else:
                    return ToolResult(True, "No matches found")
                    
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Search timed out")
        except Exception as e:
            return ToolResult(False, "", str(e))

class SystemInfoTool(BaseTool):
    """Get comprehensive system and data information."""
    
    def get_name(self) -> str:
        return "system_info"
    
    def get_description(self) -> str:
        return "Get system information, disk usage, and data overview"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_disk_usage": {
                    "type": "boolean",
                    "description": "Include disk usage information",
                    "default": True
                },
                "include_processes": {
                    "type": "boolean", 
                    "description": "Include running processes",
                    "default": False
                }
            }
        }
    
    def execute(self, include_disk_usage: bool = True, include_processes: bool = False) -> ToolResult:
        try:
            info = {}
            
            # Basic system info
            info["hostname"] = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
            info["username"] = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
            info["date"] = subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
            
            # macOS specific
            info["system_version"] = subprocess.run(["sw_vers"], capture_output=True, text=True).stdout
            
            # Disk usage
            if include_disk_usage:
                df_result = subprocess.run(["df", "-h"], capture_output=True, text=True)
                info["disk_usage"] = df_result.stdout
                
                # Specific directory sizes
                du_home = subprocess.run(["du", "-sh", "/Users/axisthornllc"], capture_output=True, text=True)
                info["home_directory_size"] = du_home.stdout.strip()
            
            # Running processes
            if include_processes:
                ps_result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                info["processes"] = ps_result.stdout
            
            # Data index info
            index_db = "/Users/axisthornllc/Documents/qwen_training/data_index.db"
            if os.path.exists(index_db):
                conn = sqlite3.connect(index_db)
                cursor = conn.execute("SELECT COUNT(*) FROM file_index")
                file_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT SUM(size) FROM file_index")
                total_size = cursor.fetchone()[0] or 0
                
                info["indexed_files"] = {
                    "count": file_count,
                    "total_size_mb": total_size / (1024 * 1024)
                }
                conn.close()
            
            output = json.dumps(info, indent=2)
            return ToolResult(True, output)
            
        except Exception as e:
            return ToolResult(False, "", str(e))