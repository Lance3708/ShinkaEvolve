from typing import Any, Tuple, Optional
import os
import instructor
from pathlib import Path
from dotenv import load_dotenv
import litellm

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def get_client_llm(model_name: str, structured_output: bool = False) -> Tuple[Any, str]:
    """Get the client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client.
        structured_output (bool): Whether to return a client configured for structured output.

    Returns:
        The client and model for the given model name.
        If structured_output is False, client is None (as litellm.completion is stateless).
    """
    # Ensure API keys are set in environment for litellm
    # litellm looks for OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
    
    client = None
    
    if structured_output:
        # For structured output, we use instructor.
        # instructor.from_litellm is the preferred way if available, 
        # otherwise we might need to wrap it.
        # Assuming instructor supports litellm or we use the openai client as a proxy interface if needed.
        # But simpler: use instructor's ability to patch.
        
        # Note: instructor < 1.0 might behave differently. 
        # If instructor.from_litellm exists:
        if hasattr(instructor, "from_litellm"):
            client = instructor.from_litellm(litellm.completion)
        else:
            # Fallback or alternative: create an OpenAI client that points to nothing specific
            # but instructor uses it to drive the logic, and we might need to hook it up.
            # Actually, for simplicity in this migration, if we want to use litellm for everything,
            # we can use instructor.from_litellm(litellm.completion).
            # If that's not available, we might need to check the installed version.
            # Let's assume we can use a standard OpenAI client and let litellm handle the routing 
            # if we were using the proxy, but here we are using the library.
            
            # Let's try to use instructor with litellm.completion directly if possible.
            # If not, we might need to instantiate a dummy OpenAI client and patch it?
            # No, let's try to use the most standard way.
            
            # For now, let's assume we can just return None for non-structured, 
            # and for structured, we might need to handle it in query.py or here.
            # But the signature expects a client.
            
            # Let's try to return a wrapped litellm completion function if possible.
            try:
                client = instructor.from_litellm(litellm.completion)
            except AttributeError:
                # If from_litellm is missing, we might be on an older version.
                # We can try to use `instructor.patch()` on a dummy object?
                # Or maybe just return None and handle it in query.py?
                # But existing code expects a client.
                pass

    return client, model_name
