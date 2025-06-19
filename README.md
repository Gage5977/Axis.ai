# AI Local Models - Consolidated Structure

## Overview
This directory contains all AI model development, training, and deployment resources organized in a clean, modular structure.

## Directory Structure

### `/mistral/`
Enhanced Mistral implementation with MSEM architecture (671B parameters)
- `models/` - Custom PyTorch implementation of Mistral 7B
- `training/` - Training scripts and pipelines
- `inference/` - Inference scripts
- `data/` - Expert-specific training data
- `configs/` - Model and training configurations
- `docs/` - Architecture and implementation documentation
- `scripts/` - Utility scripts and tools

### `/qwen/`
Qwen model implementation and resources
- `models/` - Qwen 1.5-1.8B model weights
- `training/` - Training recipes and notebooks
- `inference/` - Inference implementations
- `scripts/` - Qwen-specific scripts
- `docs/` - Documentation and guides

### `/shared/`
Common resources used across models
- `training/` - Generic training scripts
- `inference/` - Generic inference scripts
- `outputs/` - Training outputs and checkpoints
- `data/` - Shared datasets and databases
- `scripts/` - Test and utility scripts

### `/web-interfaces/`
Web-based interfaces for AI interaction
- Ollama chat interfaces
- AI chat interface
- Finance AI assistant

### `/tools/`
Utility tools and modules
- Web tools
- Data access tools
- Tool interfaces

### `/scripts/`
Main application scripts
- `ollama_web_app.py` - Ollama web application
- `local_llm_server.py` - Local LLM server
- Shell scripts for quick access

### `/environments/`
Virtual environments and dependencies (to be populated)

## Key Projects

1. **Enhanced Mistral (Priority)**
   - Standalone implementation independent of Ollama
   - MSEM architecture with 8 specialized experts
   - Advanced reasoning layer
   - 256K-2M context window
   - Target: 1200+ words/second generation

2. **Qwen Development**
   - Fine-tuning recipes
   - Custom training pipelines
   - Integration with tools

3. **Web Interfaces**
   - Multiple chat interfaces for testing
   - Tool integration capabilities

## Next Steps
1. Set up virtual environments in `/environments/`
2. Install dependencies for Mistral development
3. Begin MSEM implementation following the 18-month roadmap
4. Test inference capabilities with existing models

## Usage
See individual project directories for specific instructions.