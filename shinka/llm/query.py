from typing import List, Union, Optional, Dict, Any
import random
import os
from pydantic import BaseModel
from .client import get_client_llm
from .models.pricing import (
    CLAUDE_MODELS,
    OPENAI_MODELS,
    DEEPSEEK_MODELS,
    GEMINI_MODELS,
    ZHIPU_MODELS,
    NVIDIA_MODELS,
    BEDROCK_MODELS,
    REASONING_OAI_MODELS,
    REASONING_CLAUDE_MODELS,
    REASONING_DEEPSEEK_MODELS,
    REASONING_GEMINI_MODELS,
    REASONING_AZURE_MODELS,
    REASONING_BEDROCK_MODELS,
)
from .models.result import QueryResult
import logging
import litellm
from litellm import completion, completion_cost

logger = logging.getLogger(__name__)


THINKING_TOKENS = {
    "auto": 0,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "max": 16384,
}


def sample_batch_kwargs(
    num_samples: int,
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
    unique_filter: bool = False,
):
    """Sample a dictionary of kwargs for a given model."""
    all_kwargs = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops

    while len(all_kwargs) < num_samples and attempts < max_attempts:
        kwargs_dict = sample_model_kwargs(
            model_names=model_names,
            temperatures=temperatures,
            max_tokens=max_tokens,
            reasoning_efforts=reasoning_efforts,
            model_sample_probs=model_sample_probs,
        )

        if unique_filter:
            if kwargs_dict not in all_kwargs:
                all_kwargs.append(kwargs_dict)
        else:
            all_kwargs.append(kwargs_dict)

        attempts += 1

    if len(all_kwargs) < num_samples:
        logger.info(
            f"Could not generate {num_samples} unique kwargs combinations "
            f"after {max_attempts} attempts"
        )
        logger.info(f"Returning {len(all_kwargs)} unique kwargs combinations.")

    return all_kwargs


def sample_model_kwargs(
    model_names: Union[List[str], str] = "gpt-4o-mini-2024-07-18",
    temperatures: Union[List[float], float] = 0.0,
    max_tokens: Union[List[int], int] = 4096,
    reasoning_efforts: Union[List[str], str] = "",
    model_sample_probs: Optional[List[float]] = None,
):
    """Sample a dictionary of kwargs for a given model."""
    # Make all inputs lists
    if isinstance(model_names, str):
        model_names = [model_names]
    if isinstance(temperatures, float):
        temperatures = [temperatures]
    if isinstance(max_tokens, int):
        max_tokens = [max_tokens]
    if isinstance(reasoning_efforts, str):
        reasoning_efforts = [reasoning_efforts]

    kwargs_dict = {}
    # perform model sampling if list provided
    if model_sample_probs is not None:
        if len(model_sample_probs) != len(model_names):
            raise ValueError(
                "model_sample_probs must have the same length as model_names"
            )
        if not abs(sum(model_sample_probs) - 1.0) < 1e-9:
            raise ValueError("model_sample_probs must sum to 1")
        kwargs_dict["model_name"] = random.choices(
            model_names, weights=model_sample_probs, k=1
        )[0]
    else:
        kwargs_dict["model_name"] = random.choice(model_names)

    # perform temperature sampling if list provided
    # set temperature to 1.0 for reasoning models
    if kwargs_dict["model_name"] in (
        REASONING_OAI_MODELS
        + REASONING_CLAUDE_MODELS
        + REASONING_DEEPSEEK_MODELS
        + REASONING_GEMINI_MODELS
        + REASONING_AZURE_MODELS
        + REASONING_BEDROCK_MODELS
    ):
        kwargs_dict["temperature"] = 1.0
    else:
        kwargs_dict["temperature"] = random.choice(temperatures)

    # perform reasoning effort sampling if list provided
    # set max_completion_tokens for OAI reasoning models
    if kwargs_dict["model_name"] in (REASONING_OAI_MODELS + REASONING_AZURE_MODELS):
        kwargs_dict["max_output_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        if r_effort != "auto":
            kwargs_dict["reasoning"] = {"effort": r_effort}

    if kwargs_dict["model_name"] in (REASONING_GEMINI_MODELS):
        kwargs_dict["max_tokens"] = random.choice(max_tokens)
        r_effort = random.choice(reasoning_efforts)
        think_bool = r_effort != "auto"
        if think_bool:
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            # LiteLLM passes extra_body to the provider
            kwargs_dict["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "thinking_budget": thinking_tokens,
                        "include_thoughts": True,
                    }
                }
            }

    elif kwargs_dict["model_name"] in (
        REASONING_CLAUDE_MODELS + REASONING_BEDROCK_MODELS
    ):
        kwargs_dict["max_tokens"] = min(random.choice(max_tokens), 16384)
        r_effort = random.choice(reasoning_efforts)
        think_bool = r_effort != "auto"
        if think_bool:
            # filter thinking tokens to be smaller than max_tokens
            # not auto THINKING_TOKENS
            t = THINKING_TOKENS[r_effort]
            thinking_tokens = t if t < kwargs_dict["max_tokens"] else 1024
            # sample only from thinking tokens that are valid
            # For LiteLLM/Anthropic, this usually goes into 'thinking' param or extra_body
            # LiteLLM supports 'thinking' param for Anthropic
            kwargs_dict["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_tokens,
            }

    else:
        if (
            kwargs_dict["model_name"] in CLAUDE_MODELS
            or kwargs_dict["model_name"] in BEDROCK_MODELS
            or kwargs_dict["model_name"] in REASONING_CLAUDE_MODELS
            or kwargs_dict["model_name"] in REASONING_BEDROCK_MODELS
            or kwargs_dict["model_name"] in DEEPSEEK_MODELS
            or kwargs_dict["model_name"] in REASONING_DEEPSEEK_MODELS
        ):
            kwargs_dict["max_tokens"] = random.choice(max_tokens)
        else:
            kwargs_dict["max_output_tokens"] = random.choice(max_tokens)

    return kwargs_dict


def query(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM using LiteLLM."""
    client, model_name = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    
    if msg_history:
        messages.extend(msg_history)
        
    messages.append({"role": "user", "content": msg})

    # Map Shinka model names to LiteLLM format
    # LiteLLM needs explicit provider prefixes for some models
    litellm_model = model_name
    if model_name in GEMINI_MODELS.keys():
        # Use 'gemini/' prefix to force Google AI Studio instead of Vertex AI
        litellm_model = f"gemini/{model_name}"
    elif model_name in DEEPSEEK_MODELS.keys():
        # DeepSeek needs 'deepseek/' prefix
        litellm_model = f"deepseek/{model_name}"
    elif model_name in CLAUDE_MODELS.keys():
        # Anthropic models can use 'anthropic/' prefix
        litellm_model = f"anthropic/{model_name}"
    elif model_name in ZHIPU_MODELS.keys():
        # Zhipu models use OpenAI-compatible API
        litellm_model = model_name
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://open.bigmodel.cn/api/paas/v4/"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("ZHIPUAI_API_KEY")
    elif model_name in NVIDIA_MODELS.keys():
        # Nvidia models use OpenAI-compatible API
        litellm_model = model_name
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("NVIDIA_NIM_API_KEY")
    elif model_name.startswith("bedrock/"):
        # Bedrock models already have prefix, keep as is
        litellm_model = model_name
    elif model_name.startswith("azure-"):
        # Azure models: convert azure-<model> to azure/<model>
        litellm_model = model_name.replace("azure-", "azure/", 1)
    elif model_name.startswith("dashscope/"):
        # Dashscope models: use OpenAI-compatible API
        litellm_model = model_name.replace("dashscope/", "", 1)
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("DASHSCOPE_API_KEY")
    # OpenAI models don't need prefix usually, LiteLLM handles them by default

    # Prepare arguments for litellm
    litellm_kwargs = {
        "model": litellm_model,
        "messages": messages,
    }
    
    # Map Shinka kwargs to LiteLLM kwargs
    if "max_output_tokens" in kwargs:
        litellm_kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
    if "max_tokens" in kwargs:
        litellm_kwargs["max_tokens"] = kwargs.pop("max_tokens")
    if "temperature" in kwargs:
        litellm_kwargs["temperature"] = kwargs.pop("temperature")
    if "extra_body" in kwargs:
        litellm_kwargs["extra_body"] = kwargs.pop("extra_body")
    if "thinking" in kwargs:
        # Pass thinking directly if supported, or via extra_body
        # LiteLLM 1.40+ supports thinking param for Anthropic
        litellm_kwargs["thinking"] = kwargs.pop("thinking")
    if "reasoning" in kwargs:
        # For O1/O3 models
        if "extra_body" not in litellm_kwargs:
            litellm_kwargs["extra_body"] = {}
        litellm_kwargs["extra_body"]["reasoning_effort"] = kwargs.pop("reasoning")["effort"]

    # Add any remaining kwargs
    litellm_kwargs.update(kwargs)

    try:
        if output_model:
            # Structured output using instructor
            if client:
                # Use the instructor client
                response = client.chat.completions.create(
                    response_model=output_model,
                    **litellm_kwargs
                )
                return response
            else:
                # Fallback if no client (shouldn't happen if get_client_llm works)
                raise ValueError("Instructor client not initialized")
        else:
            # Standard completion
            response = completion(**litellm_kwargs)
            
            content = response.choices[0].message.content
            
            # Calculate cost
            try:
                cost = completion_cost(completion_response=response)
            except:
                cost = 0.0
            
            # Construct new message history
            new_msg_history = messages + [{"role": "assistant", "content": content}]
            
            return QueryResult(
                content=content,
                msg=msg,
                system_msg=system_msg,
                new_msg_history=new_msg_history,
                model_name=model_name,
                kwargs=litellm_kwargs,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=cost,
                input_cost=0.0, # LiteLLM gives total cost, splitting might be hard without pricing dict
                output_cost=0.0,
                thought="", # If thinking is enabled, we might want to extract it?
                model_posteriors=model_posteriors
            )
            
    except Exception as e:
        logger.error(f"Error in query_litellm: {e}")
        raise e


async def query_async(
    model_name: str,
    msg: str,
    system_msg: str,
    msg_history: List = [],
    output_model: Optional[BaseModel] = None,
    model_posteriors: Optional[Dict[str, float]] = None,
    **kwargs,
) -> QueryResult:
    """Query the LLM using LiteLLM asynchronously."""
    client, model_name = get_client_llm(
        model_name, structured_output=output_model is not None
    )
    
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    
    if msg_history:
        messages.extend(msg_history)
        
    messages.append({"role": "user", "content": msg})

    # Map Shinka model names to LiteLLM format
    # LiteLLM needs explicit provider prefixes for some models
    litellm_model = model_name
    if model_name in GEMINI_MODELS.keys():
        # Use 'gemini/' prefix to force Google AI Studio instead of Vertex AI
        litellm_model = f"gemini/{model_name}"
    elif model_name in DEEPSEEK_MODELS.keys():
        # DeepSeek needs 'deepseek/' prefix
        litellm_model = f"deepseek/{model_name}"
    elif model_name in CLAUDE_MODELS.keys():
        # Anthropic models can use 'anthropic/' prefix
        litellm_model = f"anthropic/{model_name}"
    elif model_name in ZHIPU_MODELS.keys():
        # Zhipu models use OpenAI-compatible API
        litellm_model = model_name
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://open.bigmodel.cn/api/paas/v4/"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("ZHIPUAI_API_KEY")
    elif model_name in NVIDIA_MODELS.keys():
        # Nvidia models use OpenAI-compatible API
        litellm_model = model_name
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://integrate.api.nvidia.com/v1"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("NVIDIA_NIM_API_KEY")
    elif model_name.startswith("bedrock/"):
        # Bedrock models already have prefix, keep as is
        litellm_model = model_name
    elif model_name.startswith("azure-"):
        # Azure models: convert azure-<model> to azure/<model>
        litellm_model = model_name.replace("azure-", "azure/", 1)
    elif model_name.startswith("dashscope/"):
        # Dashscope models: use OpenAI-compatible API
        litellm_model = model_name.replace("dashscope/", "", 1)
        if "api_base" not in kwargs:
            kwargs["api_base"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if "custom_llm_provider" not in kwargs:
            kwargs["custom_llm_provider"] = "openai"
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv("DASHSCOPE_API_KEY")
    # OpenAI models don't need prefix usually, LiteLLM handles them by default

    # Prepare arguments for litellm
    litellm_kwargs = {
        "model": litellm_model,
        "messages": messages,
    }
    
    # Map Shinka kwargs to LiteLLM kwargs
    if "max_output_tokens" in kwargs:
        litellm_kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
    if "max_tokens" in kwargs:
        litellm_kwargs["max_tokens"] = kwargs.pop("max_tokens")
    if "temperature" in kwargs:
        litellm_kwargs["temperature"] = kwargs.pop("temperature")
    if "extra_body" in kwargs:
        litellm_kwargs["extra_body"] = kwargs.pop("extra_body")
    if "thinking" in kwargs:
        # Pass thinking directly if supported, or via extra_body
        # LiteLLM 1.40+ supports thinking param for Anthropic
        litellm_kwargs["thinking"] = kwargs.pop("thinking")
    if "reasoning" in kwargs:
        # For O1/O3 models
        if "extra_body" not in litellm_kwargs:
            litellm_kwargs["extra_body"] = {}
        litellm_kwargs["extra_body"]["reasoning_effort"] = kwargs.pop("reasoning")["effort"]

    # Add any remaining kwargs
    litellm_kwargs.update(kwargs)

    try:
        if output_model:
            # Structured output using instructor with async
            if client:
                # Use the instructor client asynchronously
                # Note: instructor's async API might differ, check version compatibility
                response = await client.chat.completions.create_async(
                    response_model=output_model,
                    **litellm_kwargs
                )
                return response
            else:
                # Fallback if no client (shouldn't happen if get_client_llm works)
                raise ValueError("Instructor client not initialized")
        else:
            # Standard async completion using litellm.acompletion
            from litellm import acompletion
            
            response = await acompletion(**litellm_kwargs)
            
            content = response.choices[0].message.content
            
            # Calculate cost
            try:
                cost = completion_cost(completion_response=response)
            except:
                cost = 0.0
            
            # Construct new message history
            new_msg_history = messages + [{"role": "assistant", "content": content}]
            
            return QueryResult(
                content=content,
                msg=msg,
                system_msg=system_msg,
                new_msg_history=new_msg_history,
                model_name=model_name,
                kwargs=litellm_kwargs,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=cost,
                input_cost=0.0, # LiteLLM gives total cost, splitting might be hard without pricing dict
                output_cost=0.0,
                thought="", # If thinking is enabled, we might want to extract it?
                model_posteriors=model_posteriors
            )
            
    except Exception as e:
        logger.error(f"Error in query_async: {e}")
        raise e
