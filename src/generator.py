import os
import time
import logging
from src.configs import GenerationConfig

logger = logging.getLogger(__name__)


def _generate_openai(context: str, user_prompt: str, config: GenerationConfig, api_key: str) -> str:
    """Generate via OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nInstruction: {user_prompt}"},
        ],
        "temperature": config.temperature,
    }
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _generate_gemini(context: str, user_prompt: str, config: GenerationConfig, api_key: str) -> str:
    """Generate via Google Gemini API (google-genai SDK)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    full_prompt = f"{config.system_prompt}\n\nContext: {context}\n\nInstruction: {user_prompt}"

    gen_config = types.GenerateContentConfig(
        temperature=config.temperature,
    )
    if config.max_tokens is not None:
        gen_config.max_output_tokens = config.max_tokens

    response = client.models.generate_content(
        model=config.model,
        contents=full_prompt,
        config=gen_config,
    )
    return response.text


def _generate_anthropic(context: str, user_prompt: str, config: GenerationConfig, api_key: str) -> str:
    """Generate via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    max_tokens = config.max_tokens or 4096

    response = client.messages.create(
        model=config.model,
        max_tokens=max_tokens,
        system=config.system_prompt,
        messages=[
            {"role": "user", "content": f"Context: {context}\n\nInstruction: {user_prompt}"},
        ],
        temperature=config.temperature,
    )
    return response.content[0].text


def _resolve_api_key(provider: str) -> str | None:
    """Resolve API key from environment variables for the given provider."""
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY") or None
    elif provider == "gemini":
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None
    else:
        return os.getenv("OPENAI_API_KEY") or None


_GENERATORS = {
    "openai": _generate_openai,
    "gemini": _generate_gemini,
    "anthropic": _generate_anthropic,
}


def _parse_retry_after(exc: Exception) -> float | None:
    """
    Try to extract a retry-after value (seconds) from an API exception.

    Anthropic, OpenAI, and Google SDKs attach retry hints in different ways.
    Returns None if no hint found.
    """
    # Anthropic: e.response.headers["retry-after"]
    resp = getattr(exc, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {})
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Some SDKs put it in the message like "Please retry after 60 seconds"
    import re
    msg = str(exc)
    m = re.search(r"retry\s+after\s+(\d+)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


def generate_draft(
    context: str,
    user_prompt: str,
    config: GenerationConfig,
    api_key: str = None,
    context_limit: int = 60_000,
) -> str:
    """
    Generate a single draft using the given GenerationConfig.

    Routes to OpenAI, Gemini, or Anthropic based on config.provider.
    Returns the model response text, or a simulated placeholder when no API key is available.

    context_limit truncates the context centrally so individual provider
    functions don't need to handle it.
    """
    provider = config.provider

    # Centralized context truncation
    context = context[:context_limit]

    # Resolve key: use explicit key, then env vars
    if not api_key:
        api_key = _resolve_api_key(provider)

    if not api_key:
        return (
            f"[Simulated — no {provider.upper()} API key] Output {config.label}\n"
            f"Model: {config.model} | Temp: {config.temperature}\n"
            f"Prompt excerpt: {user_prompt[:200]}...\n"
            f"Context excerpt: {context[:300]}..."
        )

    gen_fn = _GENERATORS.get(provider)
    if gen_fn is None:
        return f"Error ({config.label}): Unknown provider '{provider}'"

    # Retry with exponential backoff for rate-limit (429) errors.
    max_retries = 4
    for attempt in range(max_retries + 1):
        try:
            return gen_fn(context, user_prompt, config, api_key)
        except Exception as e:
            err_str = str(e).lower()
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            is_rate_limit = (status == 429) or ("429" in err_str) or ("rate" in err_str and "limit" in err_str)

            if is_rate_limit and attempt < max_retries:
                # Parse retry-after hint if available, else exponential backoff
                wait = _parse_retry_after(e) or (2 ** attempt * 15)
                logger.info(
                    "Rate limited on %s (attempt %d/%d), waiting %.0fs...",
                    config.label, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue

            return f"Error ({config.label}, {config.model}): {e}"
