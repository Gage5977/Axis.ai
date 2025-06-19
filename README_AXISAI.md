# 🤖 AxisAI - Your Personal Multi-Model AI Agent

**AxisAI** is an intelligent AI agent that automatically routes your queries to the best available model based on the content and context of your question. No more thinking about which model to use - AxisAI does it for you!

## ✨ Features

- 🧠 **Intelligent Model Routing** - Automatically detects whether your question is about finance, coding, general knowledge, or needs a quick response
- 🎨 **Beautiful Terminal Interface** - Colorful and intuitive command-line experience
- 🔄 **Interactive Mode** - Chat continuously with context-aware responses
- ⚡ **Multiple Models** - Leverages Mistral 7B, QWen 14B, Finance Assistant, and Llama 3.2
- 🎛️ **Customizable** - Control temperature, model selection, and verbosity
- 🚀 **Global Access** - Available from anywhere in your terminal

## 🚀 Quick Start

### Basic Usage
```bash
# Simple question (auto-detects best model)
axisai "What is machine learning?"

# Coding question (routes to QWen 14B)
axisai "Write a Python function to sort a list"

# Finance question (routes to Finance Assistant)
axisai "Should I invest in Tesla stock?"

# Quick question (routes to Llama 3.2)
axisai -m fast "What is the capital of Japan?"
```

### Interactive Mode
```bash
# Start interactive chat
axisai -i

# In interactive mode you can:
# - Ask questions naturally
# - Type 'switch code' to change models
# - Type 'models' to see available models
# - Type 'help' for commands
# - Type 'exit' to quit
```

## 📋 Command Reference

### Options
- `-m, --model` - Specify model: `finance`, `code`, `general`, `fast`
- `-i, --interactive` - Start interactive chat mode
- `-l, --list` - List all available models
- `-t, --temp` - Set temperature (0.1-1.0, default: 0.7)
- `-v, --verbose` - Show which model is being used
- `-h, --help` - Show help message

### Examples
```bash
# Verbose mode (shows model selection)
axisai -v "Explain blockchain technology"

# Control creativity level
axisai -t 0.3 "Write a formal business email"  # More focused
axisai -t 0.9 "Write a creative story"         # More creative

# Force specific model
axisai -m code "What is recursion?"
axisai -m finance "Analyze market trends"

# List available models
axisai -l
```

## 🤖 Model Specialties

| Model | Specialty | Best For |
|-------|-----------|----------|
| **finance** | `finance-assistant:latest` | Stock analysis, investment advice, market insights |
| **code** | `qwen3:14b` | Programming, debugging, technical documentation |
| **general** | `mistral:latest` | General knowledge, explanations, creative writing |
| **fast** | `llama3.2:latest` | Quick responses, simple questions |
| **embed** | `nomic-embed-text:latest` | Text embeddings and similarity |

## 🧠 Intelligent Routing

AxisAI automatically detects your query type based on keywords:

### Finance Keywords
`stock`, `market`, `finance`, `investment`, `trading`, `portfolio`, `crypto`, `bitcoin`, `economy`, `financial`

### Code Keywords  
`code`, `program`, `function`, `debug`, `python`, `javascript`, `algorithm`, `software`, `api`, `database`

### Default
All other queries route to the **general** model (Mistral 7B)

## 🎯 Advanced Usage

### Interactive Commands
When in interactive mode (`axisai -i`):
- `switch <model>` - Change to specific model
- `models` - List available models  
- `clear` - Clear screen
- `help` - Show interactive commands
- `exit` or `quit` - Exit interactive mode

### Combining Options
```bash
# Verbose interactive mode with custom temperature
axisai -i -v -t 0.8

# Quick code question with low creativity
axisai -m code -t 0.2 "Fix this Python syntax error"
```

## 🔧 Installation

AxisAI is already installed globally on your system! Just run:
```bash
axisai --help
```

If you need to reinstall:
```bash
sudo cp axisai /usr/local/bin/
chmod +x /usr/local/bin/axisai
```

## 📊 Performance Tips

- **Fast responses**: Use `-m fast` for simple questions
- **Detailed analysis**: Let auto-routing choose the specialized model
- **Creative tasks**: Use higher temperature (`-t 0.8` or `-t 0.9`)
- **Precise tasks**: Use lower temperature (`-t 0.2` or `-t 0.3`)

## 🎨 Example Workflows

### 1. Research Assistant
```bash
axisai "Explain quantum computing in simple terms"
axisai -m code "Show me a quantum algorithm example"
axisai -m finance "What companies are investing in quantum tech?"
```

### 2. Development Helper
```bash
axisai -m code "Create a REST API in Python"
axisai -m code "Add error handling to this function"
axisai -m code "Write unit tests for this code"
```

### 3. Investment Research
```bash
axisai -m finance "Analyze AAPL stock performance"
axisai -m finance "What are the risks of investing in AI stocks?"
axisai -m finance "Compare growth vs value investing strategies"
```

## 🏆 Why AxisAI?

✅ **No Model Management** - Automatically selects the best model  
✅ **Context Aware** - Understands your question type  
✅ **Multiple Specialties** - Finance, code, general knowledge  
✅ **Fast & Efficient** - Quick responses when you need them  
✅ **Beautiful Interface** - Colorful, intuitive terminal experience  
✅ **Flexible** - Interactive mode and one-shot queries  

## 🔮 Future Enhancements

- [ ] Add more specialized models (science, medical, legal)
- [ ] Context memory across sessions
- [ ] Custom model configurations
- [ ] Integration with external APIs
- [ ] Voice input/output support

---

**Created by AxisThornLLC** - Your personal AI agent, making AI accessible through intelligent routing! 🚀 