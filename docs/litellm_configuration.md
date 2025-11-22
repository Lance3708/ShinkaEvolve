# LiteLLM Configuration and Usage Guide

## 🎯 Quick Start

> [!TIP]
> **Chinese Users**: Check the [Chinese Model Guide](../LITELLM_中国模型指南.md) (LITELLM_CHINA_MODEL_GUIDE.md) for detailed configuration of models like Qwen, Kimi, Grok, etc.

> [!NOTE]
> **Full Model List**: LiteLLM supports **95 providers and 1803 models**!
> Check the [Full List](../LITELLM_ALL_PROVIDERS.md) to see all supported models.

### Step 1: Configure API Keys

1. **Create a `.env` file**:
   ```bash
   cd /home/lance/Documents/sakana/shinka/ShinkaEvolve
   cp .env.example .env
   ```

2. **Edit the `.env` file and add your API keys**:
   ```bash
   # Add at least one provider's API key
   OPENAI_API_KEY=sk-proj-your_key
   ANTHROPIC_API_KEY=sk-ant-your_key
   DEEPSEEK_API_KEY=sk-your_key
   GEMINI_API_KEY=your_key
   ```

3. **Test API Key Configuration**:
   ```bash
   python test_api_keys.py
   ```

### Step 2: Specify Models in Experiments

In your experiment configuration file (e.g., `examples/circle_packing/run_evo.py`), find this configuration section:

```python
evo_config = EvolutionConfig(
    # ... other configs ...
    
    # Main LLM model list (modify models here)
    llm_models=[
        "gemini-2.5-flash",      # Google Gemini
        "deepseek-chat",         # DeepSeek Standard
        "deepseek-reasoner",     # DeepSeek Reasoner
    ],
    
    # LLM parameters
    llm_kwargs=dict(
        temperatures=[0.0, 0.5, 1.0],           # List of temperature values
        reasoning_efforts=["auto", "low", "medium", "high"],  # Reasoning effort
        max_tokens=8192,                        # Max tokens
    ),
    
    # Meta LLM (for summarization)
    meta_llm_models=["deepseek-chat"],
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    
    # Novelty Judge LLM (for novelty judgment)
    novelty_llm_models=["deepseek-chat"],
    novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=8192),
    
    # ... other configs ...
)
```

## 💻 Programmatic Usage

You can also use the `LLMClient` directly in your code to interact with models. Here is an example based on `tests/test_litellm_integration.py`:

```python
from shinka.llm import LLMClient

def test_model(model_name):
    print(f"\nTesting model: {model_name} ...")
    try:
        # Create a client instance for the model
        llm = LLMClient(
            model_names=model_name,
            temperatures=0.7,
            max_tokens=50,
            verbose=True
        )
        
        # Query the model
        result = llm.query(
            msg="Hello, are you working?",
            system_msg="You are a helpful assistant."
        )
        
        if result:
            print(f"✓ Model {model_name} Success!")
            print(f"  Response: {result.content}")
            return True
        else:
            print(f"✗ Model {model_name} Failed (No response)")
            return False
            
    except Exception as e:
        print(f"✗ Model {model_name} Exception: {str(e)}")
        return False

# Example usage
models_to_test = [
    "deepseek-chat",
    "gpt-4o-mini",
    "claude-3-5-haiku"
]

for model in models_to_test:
    test_model(model)
```

## 📋 Available Models List

> [!IMPORTANT]
> LiteLLM supports **95+ providers** and **1800+ models**! The following lists only common models.
> For the full list, please check [LITELLM_ALL_PROVIDERS.md](../LITELLM_ALL_PROVIDERS.md).

### 🇨🇳 Chinese Models (Recommended! High Cost-Performance)

#### Qwen (Dashscope)

```python
llm_models=["dashscope/qwen-turbo"]         # ¥0.3/¥1.2 per M tokens - Super Cheap!
llm_models=["dashscope/qwen-plus"]          # ¥2.8/¥8.4 per M tokens
llm_models=["dashscope/qwen-max"]           # ¥11.2/¥44.8 per M tokens
llm_models=["dashscope/qwen-coder"]         # Code Specialized
```

**API Key**: `DASHSCOPE_API_KEY` (Get it here: https://dashscope.console.aliyun.com/apiKey)

#### Kimi (Moonshot AI)

```python
llm_models=["moonshot/moonshot-v1-8k"]      # 8K Context
llm_models=["moonshot/moonshot-v1-32k"]     # 32K Context
llm_models=["moonshot/moonshot-v1-128k"]    # 128K Ultra-long Context
llm_models=["moonshot/kimi-thinking-preview"] # Deep Thinking
```

**API Key**: `MOONSHOT_API_KEY` (Get it here: https://platform.moonshot.cn/console/api-keys)

#### Grok (xAI)

```python
llm_models=["xai/grok-3-mini"]              # $0.3/$0.5 per M tokens
llm_models=["xai/grok-3"]                   # $3/$15 per M tokens
llm_models=["xai/grok-2-vision"]            # Image Support
```

**API Key**: `XAI_API_KEY` (Get it here: https://console.x.ai/)

#### DeepSeek

```python
llm_models=["deepseek-chat"]                # ¥1/¥2 per M tokens - Extremely High Value!
llm_models=["deepseek-reasoner"]            # ¥2/¥8 per M tokens - Reasoning Specialized
llm_models=["deepseek-v3"]                  # Latest Version
```

**API Key**: `DEEPSEEK_API_KEY` (Get it here: https://platform.deepseek.com/api_keys)

---

### 🤖 International Mainstream Models

#### OpenAI Models

```python
# Standard Models
llm_models=["gpt-4o-mini"]              # Fast, Economical ($0.15/$0.60 per M tokens)
llm_models=["gpt-4o-2024-08-06"]        # More Powerful ($2.5/$10 per M tokens)
llm_models=["gpt-4.1"]                  # Latest Version

# Reasoning Models
llm_models=["o1-2024-12-17"]            # O1 ($15/$60 per M tokens)
llm_models=["o3-mini-2025-01-31"]       # O3 Mini ($1.1/$4.4 per M tokens)
llm_models=["o3-2025-04-16"]            # O3 Full Version
```

**API Key**: `OPENAI_API_KEY=sk-proj-...`

#### Anthropic Claude Models

```python
llm_models=["claude-3-5-haiku-20241022"]     # Fast, Economical ($0.8/$4 per M tokens)
llm_models=["claude-3-5-sonnet-20241022"]    # Balanced Performance ($3/$15 per M tokens)
llm_models=["claude-3-opus-20240229"]        # Most Powerful
llm_models=["claude-3-7-sonnet-20250219"]    # With Extended Thinking
llm_models=["claude-4-sonnet-20250514"]      # Claude 4
```

**API Key**: `ANTHROPIC_API_KEY=sk-ant-...`

#### DeepSeek Models

```python
llm_models=["deepseek-chat"]         # Standard Chat ($0.27/$1.10 per M tokens)
llm_models=["deepseek-reasoner"]     # Reasoning Model ($0.55/$2.19 per M tokens)
```

**API Key**: `DEEPSEEK_API_KEY=sk-...`

#### Google Gemini Models

```python
llm_models=["gemini-2.5-pro"]                        # Pro Version ($1.25/$10 per M tokens)
llm_models=["gemini-2.5-flash"]                      # Fast Version ($0.3/$2.5 per M tokens)
llm_models=["gemini-2.5-flash-lite-preview-06-17"]   # Lite Version
llm_models=["gemini-3-pro-preview"]                  # Gemini 3 Preview
```

**API Key**: `GEMINI_API_KEY=...` or `GOOGLE_API_KEY=...`

## 🎯 Experiment Configuration Examples

### Example 1: Quick Test (Economical)

```python
evo_config = EvolutionConfig(
    # Use fast, economical models
    llm_models=[
        "gpt-4o-mini",
        "deepseek-chat",
    ],
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=4096,
    ),
    meta_llm_models=["gpt-4o-mini"],
    meta_llm_kwargs=dict(temperatures=[0.5], max_tokens=2048),
    novelty_llm_models=["gpt-4o-mini"],
    novelty_llm_kwargs=dict(temperatures=[0.3], max_tokens=2048),
)
```

**Estimated Cost**: ~$0.10-0.50 per run

### Example 2: Balanced Performance (Recommended)

```python
evo_config = EvolutionConfig(
    # Mix models with different performance characteristics
    llm_models=[
        "gpt-4o-mini",              # 50%
        "claude-3-5-haiku",         # 30%
        "deepseek-chat",            # 20%
    ],
    llm_kwargs=dict(
        temperatures=[0.5, 0.7, 0.9],
        max_tokens=8192,
    ),
    meta_llm_models=["claude-3-5-haiku"],
    meta_llm_kwargs=dict(temperatures=[0.5], max_tokens=4096),
    novelty_llm_models=["deepseek-chat"],
    novelty_llm_kwargs=dict(temperatures=[0.3], max_tokens=4096),
)
```

**Estimated Cost**: ~$1-5 per run

### Example 3: High Performance (Production)

```python
evo_config = EvolutionConfig(
    # Use high-performance models
    llm_models=[
        "claude-3-5-sonnet-20241022",
        "gpt-4o-2024-08-06",
    ],
    llm_kwargs=dict(
        temperatures=[0.6, 0.7, 0.8],
        max_tokens=8192,
    ),
    meta_llm_models=["claude-3-5-sonnet"],
    meta_llm_kwargs=dict(temperatures=[0.5], max_tokens=8192),
    novelty_llm_models=["gpt-4o"],
    novelty_llm_kwargs=dict(temperatures=[0.3], max_tokens=4096),
)
```

**Estimated Cost**: ~$5-20 per run

### Example 4: Reasoning Mode

```python
evo_config = EvolutionConfig(
    # Use reasoning models for complex problems
    llm_models=[
        "deepseek-reasoner",
        "o1-2024-12-17",
        "claude-3-7-sonnet-20250219",
    ],
    llm_kwargs=dict(
        temperatures=[1.0],          # Reasoning models fixed at 1.0
        reasoning_efforts=["medium", "high"],
        max_tokens=16384,
    ),
    meta_llm_models=["deepseek-chat"],
    meta_llm_kwargs=dict(temperatures=[0.5], max_tokens=8192),
)
```

**Estimated Cost**: ~$10-50 per run

## 🔧 Parameter Explanation

### llm_models
List of models mainly used for code generation and evolution. You can specify multiple models, and the system will select randomly.

```python
llm_models=["model1", "model2", "model3"]
```

### llm_kwargs
LLM call parameters:

- **temperatures**: List of temperature values, controlling output randomness
  - `0.0`: Deterministic output (always select the most likely token)
  - `0.5-0.7`: Balanced creativity and consistency (Recommended)
  - `0.8-1.0`: More creative
  - `1.0`: Fixed value for reasoning models

- **max_tokens**: Maximum output tokens
  - `2048-4096`: Medium output
  - `8192`: Long output
  - `16384`: Ultra-long output (Reasoning models)

- **reasoning_efforts**: Reasoning effort level (Reasoning models only)
  - `"auto"`: Automatic
  - `"low"`: Low
  - `"medium"`: Medium
  - `"high"`: High
  - `"max"`: Maximum

### meta_llm_models
Models used for meta-analysis and summarization. Usually, choose an economical but good quality model.

### novelty_llm_models
Models used for judging the novelty of solutions. Usually, choose a fast model.

## 💰 Cost Estimation and Optimization

### Cost Comparison Table

| Configuration | Main Models | Estimated Cost per Run | Use Case |
|--------------|-------------|------------------------|----------|
| Economical | gpt-4o-mini + deepseek-chat | $0.10-0.50 | Testing, Development |
| Balanced | Mix of mid-low end models | $1-5 | Standard Experiments |
| High Performance | claude-sonnet + gpt-4o | $5-20 | Important Experiments |
| Reasoning | Reasoning specialized models | $10-50 | Complex Problems |

### Cost Optimization Suggestions

1. **Use economical models during development**
   ```python
   llm_models=["gpt-4o-mini", "deepseek-chat"]
   ```

2. **Reduce max_tokens**
   ```python
   llm_kwargs=dict(max_tokens=4096)  # Instead of 8192
   ```

3. **Reduce number of generations**
   ```python
   num_generations=10  # For testing
   num_generations=100 # For production
   ```

4. **Use fewer parallel jobs**
   ```python
   max_parallel_jobs=2  # Instead of 5
   ```

## 🧪 Verifying Configuration

### Method 1: Test API Keys

```bash
python test_api_keys.py
```

This will test all configured providers and show which ones are available.

### Method 2: Simple Integration Test

```bash
python test_litellm_integration.py
```

### Method 3: Run Small-Scale Experiment

Edit `run_evo.py` to reduce the number of generations:

```python
evo_config = EvolutionConfig(
    num_generations=2,  # Temporarily reduce to 2
    max_parallel_jobs=1,
    llm_models=["gpt-4o-mini"],  # Use the cheapest model
    # ... other configs
)
```

Then run:
```bash
cd examples/circle_packing
python run_evo.py
```

## 📝 Complete Experiment Configuration Template

```python
#!/usr/bin/env python3
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# ========================================
# Step 1: Select Models to Use
# ========================================

# Option A: Economical Mode (Recommended for testing)
MAIN_MODELS = ["gpt-4o-mini", "deepseek-chat"]
META_MODELS = ["gpt-4o-mini"]
NOVELTY_MODELS = ["gpt-4o-mini"]

# Option B: Balanced Mode (Recommended for production)
# MAIN_MODELS = ["gpt-4o-mini", "claude-3-5-haiku", "deepseek-chat"]
# META_MODELS = ["claude-3-5-haiku"]
# NOVELTY_MODELS = ["deepseek-chat"]

# Option C: High Performance Mode
# MAIN_MODELS = ["claude-3-5-sonnet-20241022", "gpt-4o-2024-08-06"]
# META_MODELS = ["claude-3-5-sonnet"]
# NOVELTY_MODELS = ["gpt-4o"]

# Option D: Reasoning Mode
# MAIN_MODELS = ["deepseek-reasoner", "claude-3-7-sonnet"]
# META_MODELS = ["deepseek-chat"]
# NOVELTY_MODELS = ["deepseek-chat"]

# ========================================
# Experiment Configuration
# ========================================

evo_config = EvolutionConfig(
    # Use selected models
    llm_models=MAIN_MODELS,
    llm_kwargs=dict(
        temperatures=[0.5, 0.7, 0.9],
        reasoning_efforts=["auto", "medium", "high"],
        max_tokens=8192,
    ),
    
    meta_llm_models=META_MODELS,
    meta_llm_kwargs=dict(
        temperatures=[0.5],
        max_tokens=4096
    ),
    
    novelty_llm_models=NOVELTY_MODELS,
    novelty_llm_kwargs=dict(
        temperatures=[0.3],
        max_tokens=4096
    ),
    
    # Other configs...
    num_generations=100,
    max_parallel_jobs=5,
    # ...
)

# Run Experiment
def main():
    runner = EvolutionRunner(
        evo_config=evo_config,
        # ...
    )
    runner.run()

if __name__ == "__main__":
    main()
```

## ❓ FAQ

**Q: How do I know if my API keys are valid?**
```bash
python tests/extract_litellm_providers.py
```
