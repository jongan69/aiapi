import os
from huggingface_hub import whoami

def check_hf_inference_quota():
    """
    Check the current Hugging Face Inference API quota status.
    
    Returns a dict with:
      - total_quota: known daily call limit (int)
      - usage: number of calls used so far (None if unknown)
      - remaining_quota: remaining calls (None if unknown)
      - is_exceeded: True if usage exceeded quota (None if unknown)
    
    Note: Hugging Face provides no direct API for usage. 
    Default limits (free=1000/day, PRO=20000/day:contentReference[oaicite:5]{index=5}) are assumed.
    """
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable not set")

    # Assume default daily quotas (free vs PRO):contentReference[oaicite:6]{index=6}
    total_quota = 1000  # default free-tier daily quota
    # Detect PRO plan via whoami (if available from huggingface_hub)
    try:
        
        info = whoami(token=api_key)
        print(info)
        if info.get("subscription_type") == "PRO":
            total_quota = 20000
    except Exception:
        # huggingface_hub not installed or detection failed
        pass

    # Usage cannot be fetched programmatically; leave as None or track externally
    usage = None
    remaining = None
    is_exceeded = None
    if total_quota is not None and usage is not None:
        remaining = total_quota - usage
        is_exceeded = (remaining < 0)
    return {
        "total_quota": total_quota,
        "usage": usage,
        "remaining_quota": remaining,
        "is_exceeded": is_exceeded
    }

# Example usage:
# os.environ['HF_API_KEY'] = 'your_hf_api_token_here'
# status = check_hf_inference_quota()
# print(status)
