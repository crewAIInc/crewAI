# CrewAI Windows 11 Local LLM Setup Guide

**Turnkey setup for running CrewAI with local Ollama models on Windows 11**

Optimized for: **GTX 5080 with 64GB RAM** running Mixture-of-Experts (MoE) models

---

## 🎯 What This Setup Provides

- ✅ **No Docker required** - Native Windows installation
- ✅ **100% Local** - No API keys, no cloud dependencies, complete privacy
- ✅ **Turnkey installation** - Automated scripts handle everything
- ✅ **MoE Models** - Qwen2.5, DeepSeek-R1, Microsoft Phi4
- ✅ **GPU Optimized** - Configured for your GTX 5080
- ✅ **Multiple Models** - Different models for different agent roles
- ✅ **Production Ready** - Proper virtual environment and dependencies

---

## 📋 Prerequisites

### Required
- **Windows 11** (64-bit)
- **Python 3.10, 3.11, 3.12, or 3.13** ([Download](https://www.python.org/downloads/))
  - ⚠️ During installation, check "Add Python to PATH"
- **NVIDIA GPU** with updated drivers (for GTX 5080)
- **50-100GB free disk space** (for models)
- **Internet connection** (for initial setup only)

### Recommended
- **Git for Windows** ([Download](https://git-scm.com/download/win))
- **Windows Terminal** for better console experience

---

## 🚀 Quick Start (5 Steps)

### Step 1: Install Ollama

1. Download Ollama for Windows: https://ollama.ai/download
2. Run the installer
3. Ollama will install and start automatically

Verify installation:
```cmd
ollama --version
```

### Step 2: Clone/Download This Repository

If you have Git:
```cmd
git clone https://github.com/crewAIInc/crewAI.git
cd crewAI
```

Or download and extract the ZIP from GitHub.

### Step 3: Run Windows Setup Script

**Double-click `setup_windows.bat`** or run from Command Prompt:

```cmd
setup_windows.bat
```

This will:
- ✅ Check Python version
- ✅ Create virtual environment (`venv/`)
- ✅ Install CrewAI and dependencies
- ✅ Configure environment variables
- ✅ Verify Ollama installation

**Time: ~5-10 minutes** depending on internet speed

### Step 4: Install AI Models

**Double-click `install_models.bat`** or run:

```cmd
install_models.bat
```

**Recommended models** for your GTX 5080:

| Model | Size | Best For | Download Time* |
|-------|------|----------|----------------|
| **qwen2.5:32b** | ~20GB | Deep reasoning, research | ~10 min |
| **deepseek-r1:14b** | ~8GB | Fast reasoning, analysis | ~5 min |
| **phi4:14b** | ~8GB | Balanced, content writing | ~5 min |
| **llama3.2:3b** | ~2GB | Quick tasks, validation | ~2 min |

*On 100 Mbps connection

**Total download: ~38GB for all 4 models**

You can install all at once or select individual models.

### Step 5: Run the Example

**Double-click `run_example.bat`** or run:

```cmd
run_example.bat
```

This will:
1. Activate the virtual environment
2. Check Ollama connection
3. Run a multi-agent research crew with your topic
4. Save output to `windows_local_example/output/`

**First run will be slower** as models load into GPU memory.

---

## 📁 What Got Installed

```
crewAI/
├── venv/                          # Python virtual environment
├── .env                           # Your local configuration (auto-created)
├── .env.example                   # Configuration template
│
├── setup_windows.bat              # Main setup script
├── install_models.bat             # Model installer
├── run_example.bat                # Run demo crew
├── test_ollama.bat                # Test Ollama setup
├── start_ollama.bat               # Start Ollama manually
│
├── windows_local_example/         # Example crew
│   ├── main.py                    # Example script
│   ├── config/
│   │   ├── agents.yaml            # Agent configurations
│   │   └── tasks.yaml             # Task configurations
│   └── output/                    # Results saved here
│
└── WINDOWS_SETUP.md               # This file
```

---

## 🎮 Using Different Models

### Model Comparison for Your GTX 5080

| Model | Parameters | VRAM | Speed | Reasoning | Writing | Best Use Case |
|-------|-----------|------|-------|-----------|---------|---------------|
| **Qwen2.5:32b** | 32B | ~20GB | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Complex problem-solving |
| **DeepSeek-R1:14b** | 14B | ~9GB | Fast | ⭐⭐⭐⭐ | ⭐⭐⭐ | Data analysis, code |
| **Phi4:14b** | 14B | ~9GB | Fast | ⭐⭐⭐ | ⭐⭐⭐⭐ | Content generation |
| **Llama3.2:3b** | 3B | ~2GB | Very Fast | ⭐⭐ | ⭐⭐⭐ | Quick validation |
| **Qwen2.5:14b** | 14B | ~9GB | Fast | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced alternative |
| **DeepSeek-R1:32b** | 32B | ~20GB | Medium | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced reasoning |
| **Llama3.3:70b** | 70B | ~40GB† | Slow | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Maximum capability |

†70B model will use quantization on your GPU

### How to Switch Models

**Option 1: Edit .env file**
```env
MODEL=ollama/qwen2.5:32b
```

**Option 2: Specify in agents.yaml**
```yaml
researcher:
  llm: ollama/deepseek-r1:32b
```

**Option 3: In Python code**
```python
from crewai import Agent, LLM

agent = Agent(
    role="Researcher",
    llm=LLM(model="ollama/qwen2.5:32b"),
    # ... other config
)
```

### Pull Additional Models

```cmd
ollama pull qwen2.5:14b
ollama pull llama3.3:70b
ollama pull deepseek-r1:32b
```

List installed models:
```cmd
ollama list
```

---

## 🔧 Configuration

### .env Configuration

The `.env` file controls default settings:

```env
# Main model (used when agents don't specify their own)
MODEL=ollama/qwen2.5:32b

# Ollama endpoint
OLLAMA_API_BASE=http://localhost:11434

# Enable debug logging
CREWAI_DEBUG=0

# Disable telemetry (optional)
CREWAI_TELEMETRY_OPT_OUT=true
```

### Agent Configuration (agents.yaml)

```yaml
researcher:
  role: Senior Research Analyst
  goal: Uncover cutting-edge developments
  backstory: Expert researcher...
  llm: ollama/qwen2.5:32b      # Specify model per agent
  verbose: true
  max_iter: 15
  max_rpm: 0                    # No rate limit for local!
```

### Memory and Performance

For your 64GB RAM system, you can run **multiple models simultaneously**:

```yaml
# Example: Different models for different agents
researcher:
  llm: ollama/qwen2.5:32b       # 20GB VRAM

analyst:
  llm: ollama/deepseek-r1:14b   # 9GB VRAM

writer:
  llm: ollama/phi4:14b          # 9GB VRAM

# Total: ~38GB VRAM - well within GTX 5080 capacity
```

---

## 🐛 Troubleshooting

### Issue: "Ollama service is not responding"

**Solution:**
```cmd
# Start Ollama manually
ollama serve

# Or double-click start_ollama.bat
```

### Issue: "Model not found"

**Solution:**
```cmd
# Pull the model
ollama pull qwen2.5:32b

# Verify it's installed
ollama list
```

### Issue: "Out of memory" or slow performance

**Solutions:**

1. **Close other GPU applications** (browsers with hardware acceleration, games, etc.)

2. **Use smaller models:**
   ```env
   MODEL=ollama/qwen2.5:14b  # Instead of 32b
   ```

3. **Reduce concurrent agents** - run tasks sequentially instead of parallel

4. **Enable quantization** (already default for Ollama)

### Issue: "Virtual environment not found"

**Solution:**
```cmd
# Run setup again
setup_windows.bat
```

### Issue: Models are slow on first run

**This is normal!** First run loads the model into GPU memory. Subsequent runs are much faster.

**Tip:** Keep Ollama running in background:
```cmd
start /B ollama serve
```

### Issue: Python not found

**Solution:**
1. Install Python from https://www.python.org/downloads/
2. ⚠️ **Check "Add Python to PATH" during installation**
3. Restart Command Prompt
4. Run `setup_windows.bat` again

### Issue: Scripts won't run / permission denied

**Solution:**
```cmd
# Run Command Prompt as Administrator
# Right-click > "Run as administrator"
```

---

## 🧪 Testing Your Setup

Run the test script to verify everything:

```cmd
test_ollama.bat
```

This checks:
- ✅ Ollama installation
- ✅ Service is running
- ✅ Models are installed
- ✅ Model can respond to prompts

---

## 📊 Performance Tips for GTX 5080

### 1. Keep Models in Memory

Once loaded, models stay in GPU memory for fast subsequent calls:

```cmd
# Pre-load a model
ollama run qwen2.5:32b "test"
```

### 2. Batch Processing

Process multiple items in one crew run instead of multiple runs:

```python
topics = ["AI", "Blockchain", "Quantum Computing"]
for topic in topics:
    result = crew.kickoff(inputs={"topic": topic})
```

### 3. Use Model Mixing

Assign expensive models to complex tasks, fast models to simple tasks:

```yaml
researcher: ollama/qwen2.5:32b      # Complex reasoning
reviewer: ollama/llama3.2:3b        # Quick validation
```

### 4. Optimize Context Windows

Limit context to what's needed:

```python
agent = Agent(
    role="Writer",
    max_iter=10,  # Limit iterations
    # ...
)
```

### 5. Monitor GPU Usage

```cmd
# Windows Task Manager > Performance > GPU
# Or use nvidia-smi in WSL/Linux
```

---

## 🔄 Switching to VLLM (Advanced)

While Ollama is recommended for simplicity, you can also use VLLM:

### Install VLLM in WSL2

```bash
# In WSL2 Ubuntu
pip install vllm
```

### Run VLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

### Configure CrewAI to use VLLM

```env
# In .env
MODEL=openai/Qwen2.5-32B-Instruct
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy  # VLLM doesn't require real key
```

**Note:** VLLM offers better throughput for high-concurrency scenarios but is more complex to set up.

---

## 📚 Next Steps

### Create Your Own Crew

1. **Copy the example:**
   ```cmd
   xcopy windows_local_example my_crew /E /I
   ```

2. **Edit `config/agents.yaml`** - Define your agents

3. **Edit `config/tasks.yaml`** - Define your tasks

4. **Edit `main.py`** - Customize the workflow

5. **Run it:**
   ```cmd
   cd my_crew
   ..\venv\Scripts\python main.py
   ```

### Explore CrewAI Features

- **Tools:** Add web search, file operations, code execution
- **Memory:** Enable long-term memory for agents
- **RAG:** Use knowledge bases with embeddings
- **Parallel Processing:** Run multiple crews concurrently

See full documentation: https://docs.crewai.com

### Join the Community

- **Discord:** https://discord.com/invite/X4JWnZnxPb
- **GitHub:** https://github.com/crewAIInc/crewAI
- **Docs:** https://docs.crewai.com

---

## 🆘 Getting Help

### If something doesn't work:

1. **Run diagnostics:**
   ```cmd
   test_ollama.bat
   ```

2. **Check logs:**
   - Ollama logs: `%LOCALAPPDATA%\Ollama\logs`
   - CrewAI verbose output: Set `CREWAI_DEBUG=1` in `.env`

3. **Common fixes:**
   - Restart Ollama: `start_ollama.bat`
   - Restart Command Prompt
   - Re-run setup: `setup_windows.bat`
   - Reinstall Ollama

4. **Ask for help:**
   - CrewAI Discord: https://discord.com/invite/X4JWnZnxPb
   - GitHub Issues: https://github.com/crewAIInc/crewAI/issues
   - Ollama Discord: https://discord.gg/ollama

---

## 📖 Additional Resources

### Ollama
- Website: https://ollama.ai
- Models Library: https://ollama.ai/library
- GitHub: https://github.com/ollama/ollama

### CrewAI
- Documentation: https://docs.crewai.com
- Examples: https://github.com/crewAIInc/crewAI-examples
- Tools: https://github.com/crewAIInc/crewAI-tools

### Models
- Qwen: https://huggingface.co/Qwen
- DeepSeek: https://huggingface.co/deepseek-ai
- Phi-4: https://huggingface.co/microsoft/phi-4
- Llama: https://huggingface.co/meta-llama

---

## ⚡ Quick Reference

```cmd
# Setup
setup_windows.bat              # Initial setup
install_models.bat             # Download models
test_ollama.bat                # Test installation

# Running
run_example.bat                # Run demo crew
start_ollama.bat               # Start Ollama service

# Ollama Commands
ollama list                    # List installed models
ollama pull MODEL              # Download a model
ollama rm MODEL                # Remove a model
ollama run MODEL               # Test a model interactively
ollama serve                   # Start Ollama server

# Python (in activated venv)
venv\Scripts\activate          # Activate environment
python main.py                 # Run your crew
deactivate                     # Exit environment
```

---

## 🎉 You're All Set!

Your Windows 11 system is now configured to run CrewAI with powerful local MoE models.

**No cloud dependencies. No API costs. Complete control.**

Happy building! 🚀

---

*Last updated: 2025-01-18*
*Optimized for: Windows 11, GTX 5080, 64GB RAM*
