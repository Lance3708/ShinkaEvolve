#!/usr/bin/env python3
"""
Extract all LiteLLM supported providers and model information using the installed litellm library.
"""

import litellm
import json
from collections import defaultdict
import os
from pathlib import Path

def get_env_key_name(provider):
    """Infer environment variable name from provider name."""
    # Common mappings
    env_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
        'google': 'GEMINI_API_KEY',
        'vertex_ai': 'VERTEX_AI_PROJECT', # and VERTEX_AI_LOCATION
        'cohere': 'COHERE_API_KEY',
        'replicate': 'REPLICATE_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'ai21': 'AI21_API_KEY',
        'azure': 'AZURE_API_KEY', # and AZURE_API_BASE
        'bedrock': 'AWS_ACCESS_KEY_ID', # and AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
        'sagemaker': 'AWS_ACCESS_KEY_ID',
        'deepseek': 'DEEPSEEK_API_KEY',
        'groq': 'GROQ_API_KEY',
        'mistral': 'MISTRAL_API_KEY',
        'together_ai': 'TOGETHERAI_API_KEY',
        'perplexity': 'PERPLEXITYAI_API_KEY',
        'fireworks_ai': 'FIREWORKS_AI_API_KEY',
        'deepinfra': 'DEEPINFRA_API_KEY',
        'ollama': 'OLLAMA_API_BASE',
        'huggingface': 'HUGGINGFACE_API_KEY',
        'cloudflare': 'CLOUDFLARE_API_KEY',
        'voyage': 'VOYAGE_API_KEY',
        'databricks': 'DATABRICKS_API_KEY',
        'watsonx': 'WATSONX_API_KEY',
        'xai': 'XAI_API_KEY',
        'dashscope': 'DASHSCOPE_API_KEY',
        'volcengine': 'VOLCENGINE_API_KEY',
        'moonshot': 'MOONSHOT_API_KEY',
        'sambanova': 'SAMBANOVA_API_KEY',
        'cerebras': 'CEREBRAS_API_KEY',
        'nvidia_nim': 'NVIDIA_NIM_API_KEY',
        'hyperbolic': 'HYPERBOLIC_API_KEY',
        'snowflake': 'SNOWFLAKE_API_KEY',
        'friendliai': 'FRIENDLIAI_API_KEY',
        'abacus': 'ABACUS_API_KEY',
        'anyscale': 'ANYSCALE_API_KEY',
        'baichuan': 'BAICHUAN_API_KEY',
        'minimax': 'MINIMAX_API_KEY',
        'yi': 'YI_API_KEY',
        'zhipu': 'ZHIPUAI_API_KEY',
        '01': 'ZERO_ONE_API_KEY',
        'nlp_cloud': 'NLP_CLOUD_API_KEY',
        'petals': 'PETALS_API_BASE',
        'aleph_alpha': 'ALEPH_ALPHA_API_KEY',
        'predibase': 'PREDIBASE_API_KEY',
        'codestral': 'CODESTRAL_API_KEY',
        'text-completion-codestral': 'CODESTRAL_API_KEY',
    }
    
    return env_mapping.get(provider, f"{provider.upper()}_API_KEY")

def main():
    print("Extracting LiteLLM supported providers and models...")
    
    if not hasattr(litellm, 'model_cost'):
        print("Error: litellm.model_cost not found.")
        return

    model_cost = litellm.model_cost
    
    # Group models by provider
    provider_models = defaultdict(list)
    
    for model_name, info in model_cost.items():
        provider = info.get('litellm_provider', 'unknown')
        # Some models might not have provider set, try to infer from name
        if provider == 'unknown':
            if '/' in model_name:
                provider = model_name.split('/')[0]
            else:
                provider = 'openai' # Default assumption
        
        provider_models[provider].append((model_name, info))

    # Manual additions for missing providers/models
    # Verified via tests/verify_models.py
    manual_providers = {
        'zhipu': {
            'env_var': 'ZHIPUAI_API_KEY',
            'models': [
                ('glm-4', {'mode': 'chat', 'input_cost_per_token': 0.1/1e6, 'output_cost_per_token': 0.1/1e6, 'max_tokens': 128000}),
                ('glm-4-plus', {'mode': 'chat', 'input_cost_per_token': 0.1/1e6, 'output_cost_per_token': 0.1/1e6, 'max_tokens': 128000}),
                ('glm-4-0520', {'mode': 'chat', 'input_cost_per_token': 0.1/1e6, 'output_cost_per_token': 0.1/1e6, 'max_tokens': 128000}),
                ('glm-4-air', {'mode': 'chat', 'input_cost_per_token': 0.001/1e3, 'output_cost_per_token': 0.001/1e3, 'max_tokens': 128000}),
                ('glm-4-airx', {'mode': 'chat', 'input_cost_per_token': 0.001/1e3, 'output_cost_per_token': 0.001/1e3, 'max_tokens': 128000}),
                ('glm-4-flash', {'mode': 'chat', 'input_cost_per_token': 0, 'output_cost_per_token': 0, 'max_tokens': 128000}),
                ('glm-4-long', {'mode': 'chat', 'input_cost_per_token': 0.001/1e3, 'output_cost_per_token': 0.001/1e3, 'max_tokens': 1000000}),
                ('glm-4.5', {'mode': 'chat', 'input_cost_per_token': 0.1/1e6, 'output_cost_per_token': 0.1/1e6, 'max_tokens': 128000}),
                ('glm-3-turbo', {'mode': 'chat', 'input_cost_per_token': 0.001/1e3, 'output_cost_per_token': 0.002/1e3, 'max_tokens': 128000}),
            ]
        },
        'nvidia_nim': {
            'env_var': 'NVIDIA_NIM_API_KEY',
            'models': [
                ('meta/llama3-8b-instruct', {'mode': 'chat'}),
                ('mistralai/mixtral-8x22b-instruct-v0.1', {'mode': 'chat'}),
                ('google/gemma-2-9b-it', {'mode': 'chat'}),
                ('google/gemma-2-27b-it', {'mode': 'chat'}),
                ('microsoft/phi-3-medium-4k-instruct', {'mode': 'chat'}),
            ]
        }
    }

    # Merge manual providers into provider_models
    for provider, data in manual_providers.items():
        # If provider already exists, append unique models
        existing_models = {m[0] for m in provider_models.get(provider, [])}
        for model_name, info in data['models']:
            if model_name not in existing_models:
                provider_models[provider].append((model_name, info))

    # Get all known providers from litellm
    all_known_providers = set()
    if hasattr(litellm, 'provider_list'):
        for p in litellm.provider_list:
             # p is an enum or string
             if hasattr(p, 'value'):
                 all_known_providers.add(p.value)
             else:
                 all_known_providers.add(str(p))
    
    # Add providers found in model_cost and manual list
    all_known_providers.update(provider_models.keys())
    all_known_providers.update(manual_providers.keys())
    
    sorted_providers = sorted(all_known_providers)
    
    output = []
    output.append("# LiteLLM Supported Providers and Models\n")
    output.append(f"Total Providers: {len(sorted_providers)}\n")
    output.append(f"Total Models with Pricing/Info: {sum(len(m) for m in provider_models.values())}\n\n")
    
    for provider in sorted_providers:
        # Use manual env var if available, otherwise infer
        if provider in manual_providers:
            env_var = manual_providers[provider]['env_var']
        else:
            env_var = get_env_key_name(provider)
            
        models = provider_models.get(provider, [])
        
        output.append(f"## Provider: {provider}\n")
        output.append(f"- **Environment Variable**: `{env_var}`\n")
        output.append(f"- **Total Models**: {len(models)}\n")
        
        if not models:
            output.append("\n*(No specific models found in model_cost registry for this provider)*\n")
            output.append("\n---\n")
            continue

        # Separate models by mode
        chat_models = []
        other_models = []
        
        for model_name, info in models:
            mode = info.get('mode', 'chat') # Default to chat if not specified
            # Check for traditional natural language input support
            if mode in ['chat', 'completion']:
                chat_models.append((model_name, info))
            else:
                other_models.append((model_name, info))
        
        if chat_models:
            output.append("\n### Chat/Completion Models\n")
            output.append("| Model Name | Input Price ($/M) | Output Price ($/M) | Max Tokens |\n")
            output.append("|------------|-------------------|--------------------|------------|\n")
            
            for model_name, info in sorted(chat_models, key=lambda x: x[0]):
                input_cost = info.get('input_cost_per_token', 0) or 0
                output_cost = info.get('output_cost_per_token', 0) or 0
                max_tokens = info.get('max_tokens', info.get('max_output_tokens', 'N/A'))
                
                # Convert to per Million tokens
                input_price_m = input_cost * 1_000_000
                output_price_m = output_cost * 1_000_000
                
                output.append(f"| `{model_name}` | ${input_price_m:.4f} | ${output_price_m:.4f} | {max_tokens} |\n")
        
        if other_models:
            output.append("\n### Other Models (Non-Standard NL Input)\n")
            output.append("| Model Name | Mode | Pricing Info |\n")
            output.append("|------------|------|--------------|\n")
            
            for model_name, info in sorted(other_models, key=lambda x: x[0]):
                mode = info.get('mode', 'unknown')
                input_cost = info.get('input_cost_per_token', 0) or 0
                output_cost = info.get('output_cost_per_token', 0) or 0
                
                pricing = "N/A"
                if input_cost or output_cost:
                    pricing = f"In: ${input_cost*1e6:.4f}/M, Out: ${output_cost*1e6:.4f}/M"
                    
                output.append(f"| `{model_name}` | {mode} | {pricing} |\n")
        
        output.append("\n---\n")

    output_file = Path("/home/lance/Documents/sakana/shinka/ShinkaEvolve/LITELLM_ALL_PROVIDERS.md")
    output_file.write_text(''.join(output), encoding='utf-8')
    print(f"Report saved to {output_file.absolute()}")

if __name__ == "__main__":
    main()
