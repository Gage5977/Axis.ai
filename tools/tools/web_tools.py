#!/usr/bin/env python3
"""
Web access and search tools for Qwen model.
Provides web search, scraping, and API access capabilities.
"""

import requests
import json
import urllib.parse
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import re
from datetime import datetime

from .tool_interface import BaseTool, ToolResult

class WebSearchTool(BaseTool):
    """Search the web using multiple search engines."""
    
    def get_name(self) -> str:
        return "web_search"
    
    def get_description(self) -> str:
        return "Search the web using DuckDuckGo or other search engines"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10
                },
                "search_engine": {
                    "type": "string",
                    "description": "Search engine to use",
                    "enum": ["duckduckgo", "searx"],
                    "default": "duckduckgo"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, num_results: int = 10, search_engine: str = "duckduckgo") -> ToolResult:
        try:
            if search_engine == "duckduckgo":
                return self._search_duckduckgo(query, num_results)
            elif search_engine == "searx":
                return self._search_searx(query, num_results)
            else:
                return ToolResult(False, "", f"Unsupported search engine: {search_engine}")
                
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _search_duckduckgo(self, query: str, num_results: int) -> ToolResult:
        """Search using DuckDuckGo instant answers API."""
        try:
            # DuckDuckGo instant answers
            instant_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(instant_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Abstract/instant answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Instant Answer"),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "type": "instant_answer"
                })
            
            # Related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Result", "").split(" - ")[0] if " - " in topic.get("Result", "") else "Related",
                        "snippet": topic["Text"],
                        "url": topic.get("FirstURL", ""),
                        "type": "related_topic"
                    })
            
            # If no instant results, try HTML scraping
            if not results:
                results = self._scrape_duckduckgo_html(query, num_results)
            
            if results:
                output = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results[:num_results], 1):
                    output += f"{i}. {result['title']}\n"
                    output += f"   {result['snippet']}\n"
                    if result['url']:
                        output += f"   URL: {result['url']}\n"
                    output += "\n"
                
                return ToolResult(True, output, metadata={"count": len(results), "query": query})
            else:
                return ToolResult(True, f"No results found for '{query}'")
                
        except Exception as e:
            return ToolResult(False, "", str(e))
    
    def _scrape_duckduckgo_html(self, query: str, num_results: int) -> List[Dict]:
        """Scrape DuckDuckGo HTML results as fallback."""
        try:
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result_div in soup.find_all('div', class_='result')[:num_results]:
                title_link = result_div.find('a', class_='result__a')
                snippet_div = result_div.find('a', class_='result__snippet')
                
                if title_link:
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "type": "web_result"
                    })
            
            return results
            
        except Exception:
            return []
    
    def _search_searx(self, query: str, num_results: int) -> ToolResult:
        """Search using public SearX instance."""
        try:
            searx_url = "https://searx.be/search"
            params = {
                "q": query,
                "format": "json",
                "categories": "general"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; QwenBot/1.0)"
            }
            
            response = requests.get(searx_url, params=params, headers=headers, timeout=15)
            data = response.json()
            
            results = []
            for result in data.get("results", [])[:num_results]:
                results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("content", ""),
                    "url": result.get("url", ""),
                    "type": "web_result"
                })
            
            if results:
                output = f"Search results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    output += f"{i}. {result['title']}\n"
                    output += f"   {result['snippet']}\n"
                    output += f"   URL: {result['url']}\n\n"
                
                return ToolResult(True, output, metadata={"count": len(results), "query": query})
            else:
                return ToolResult(True, f"No results found for '{query}'")
                
        except Exception as e:
            return ToolResult(False, "", str(e))

class WebScrapeTool(BaseTool):
    """Scrape content from web pages."""
    
    def get_name(self) -> str:
        return "web_scrape"
    
    def get_description(self) -> str:
        return "Extract text content from web pages"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to scrape"
                },
                "extract_links": {
                    "type": "boolean",
                    "description": "Extract all links from the page",
                    "default": False
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract image URLs",
                    "default": False
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length",
                    "default": 10000
                }
            },
            "required": ["url"]
        }
    
    def execute(self, url: str, extract_links: bool = False, extract_images: bool = False, max_length: int = 10000) -> ToolResult:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            
            result = {
                "url": url,
                "title": soup.title.string if soup.title else "No title",
                "content": text,
                "content_length": len(text)
            }
            
            # Extract links if requested
            if extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    link_url = link['href']
                    if link_url.startswith('http'):
                        links.append({
                            "text": link.get_text(strip=True),
                            "url": link_url
                        })
                result["links"] = links[:50]  # Limit to 50 links
            
            # Extract images if requested
            if extract_images:
                images = []
                for img in soup.find_all('img', src=True):
                    img_url = img['src']
                    if img_url.startswith('http'):
                        images.append({
                            "alt": img.get('alt', ''),
                            "url": img_url
                        })
                result["images"] = images[:20]  # Limit to 20 images
            
            output = f"Scraped content from: {url}\n"
            output += f"Title: {result['title']}\n"
            output += f"Content length: {result['content_length']} characters\n\n"
            output += f"Content:\n{result['content']}\n"
            
            if extract_links and result.get('links'):
                output += f"\nFound {len(result['links'])} links:\n"
                for link in result['links'][:10]:
                    output += f"- {link['text']}: {link['url']}\n"
            
            if extract_images and result.get('images'):
                output += f"\nFound {len(result['images'])} images:\n"
                for img in result['images'][:10]:
                    output += f"- {img['alt']}: {img['url']}\n"
            
            return ToolResult(True, output, metadata=result)
            
        except Exception as e:
            return ToolResult(False, "", str(e))

class NewsSearchTool(BaseTool):
    """Search for recent news articles."""
    
    def get_name(self) -> str:
        return "news_search"
    
    def get_description(self) -> str:
        return "Search for recent news articles on specific topics"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "News search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10
                },
                "language": {
                    "type": "string",
                    "description": "Language preference",
                    "default": "en"
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, num_results: int = 10, language: str = "en") -> ToolResult:
        try:
            # Use DuckDuckGo news search
            search_url = "https://duckduckgo.com/"
            params = {
                "q": f"{query} news",
                "iar": "news",
                "df": "w"  # Past week
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            news_articles = soup.find_all('div', class_='news-result')[:num_results]
            
            for article in news_articles:
                title_elem = article.find('a')
                snippet_elem = article.find('span', class_='news-result__snippet')
                source_elem = article.find('span', class_='news-result__source')
                date_elem = article.find('span', class_='news-result__timestamp')
                
                if title_elem:
                    results.append({
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get('href', ''),
                        "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                        "source": source_elem.get_text(strip=True) if source_elem else "",
                        "date": date_elem.get_text(strip=True) if date_elem else ""
                    })
            
            if results:
                output = f"News results for '{query}':\n\n"
                for i, result in enumerate(results, 1):
                    output += f"{i}. {result['title']}\n"
                    output += f"   Source: {result['source']}\n"
                    if result['date']:
                        output += f"   Date: {result['date']}\n"
                    output += f"   {result['snippet']}\n"
                    output += f"   URL: {result['url']}\n\n"
                
                return ToolResult(True, output, metadata={"count": len(results), "query": query})
            else:
                return ToolResult(True, f"No recent news found for '{query}'")
                
        except Exception as e:
            return ToolResult(False, "", str(e))

class WebMonitorTool(BaseTool):
    """Monitor websites for changes."""
    
    def get_name(self) -> str:
        return "web_monitor"
    
    def get_description(self) -> str:
        return "Check website status and monitor for changes"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to monitor"
                },
                "check_content": {
                    "type": "boolean",
                    "description": "Check for content changes",
                    "default": False
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 10
                }
            },
            "required": ["urls"]
        }
    
    def execute(self, urls: List[str], check_content: bool = False, timeout: int = 10) -> ToolResult:
        try:
            results = []
            
            for url in urls:
                try:
                    start_time = datetime.now()
                    response = requests.get(url, timeout=timeout)
                    end_time = datetime.now()
                    
                    response_time = (end_time - start_time).total_seconds()
                    
                    result = {
                        "url": url,
                        "status_code": response.status_code,
                        "response_time": response_time,
                        "accessible": response.status_code == 200,
                        "content_length": len(response.text),
                        "headers": dict(response.headers)
                    }
                    
                    if check_content and response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title = soup.title.string if soup.title else "No title"
                        result["title"] = title
                        result["content_preview"] = response.text[:500]
                    
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "status_code": None,
                        "accessible": False,
                        "error": str(e)
                    })
            
            # Format output
            output = f"Website monitoring results for {len(urls)} URLs:\n\n"
            
            for result in results:
                output += f"URL: {result['url']}\n"
                if result['accessible']:
                    output += f"  Status: ✓ Online (HTTP {result['status_code']})\n"
                    output += f"  Response time: {result['response_time']:.2f}s\n"
                    output += f"  Content length: {result['content_length']} bytes\n"
                    if 'title' in result:
                        output += f"  Title: {result['title']}\n"
                else:
                    output += f"  Status: ✗ Offline\n"
                    if 'error' in result:
                        output += f"  Error: {result['error']}\n"
                output += "\n"
            
            return ToolResult(True, output, metadata={"results": results})
            
        except Exception as e:
            return ToolResult(False, "", str(e))