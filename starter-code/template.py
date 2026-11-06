"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.300},
    "gemini-3.1-flash-lite": {"input": 0.075, "output": 0.300},
    "gemini-3.5-flash-lite": {"input": 0.075, "output": 0.300},
    "gemini-3.6-flash": {"input": 0.075, "output": 0.300},
    "gemini-3.8-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_3_1_LITE_MODEL = "gemini-3.1-flash-lite"
GEMINI_3_5_LITE_MODEL = "gemini-3.5-flash-lite"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    import openai

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    response_text = getattr(response, "text", "") or ""
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    response_text = ""
    if hasattr(response, "content") and response.content:
        first_part = response.content[0]
        response_text = getattr(first_part, "text", "") or ""

    input_tokens = 0
    output_tokens = 0
    if hasattr(response, "usage") and response.usage is not None:
        input_tokens = getattr(response.usage, "input_tokens", 0) or 0
        output_tokens = getattr(response.usage, "output_tokens", 0) or 0

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str = "") -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    def _calculate_cost(model_key: str, in_tokens: int, out_tokens: int) -> float:
        rates = PRICING_1M_TOKENS.get(model_key, {"input": 0.0, "output": 0.0})
        return (in_tokens * rates["input"] + out_tokens * rates["output"]) / 1_000_000.0

    try:
        gpt4o_text, gpt4o_lat, gpt4o_usage = call_openai(prompt, model=OPENAI_MODEL)
    except Exception as e:
        gpt4o_text, gpt4o_lat, gpt4o_usage = f"[OpenAI Error: {e}]", 0.0, {"input_tokens": 0, "output_tokens": 0}

    try:
        mini_text, mini_lat, mini_usage = call_openai(prompt, model=OPENAI_MINI_MODEL)
    except Exception as e:
        mini_text, mini_lat, mini_usage = f"[OpenAI Error: {e}]", 0.0, {"input_tokens": 0, "output_tokens": 0}

    gemini_text, gemini_lat, gemini_usage = call_gemini(prompt, model=GEMINI_MODEL)

    return {
        "gpt4o": {
            "response": gpt4o_text,
            "latency": gpt4o_lat,
            "cost": _calculate_cost(OPENAI_MODEL, gpt4o_usage["input_tokens"], gpt4o_usage["output_tokens"]),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"],
        },
        "gpt4o_mini": {
            "response": mini_text,
            "latency": mini_lat,
            "cost": _calculate_cost(OPENAI_MINI_MODEL, mini_usage["input_tokens"], mini_usage["output_tokens"]),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"],
        },
        "gemini_flash": {
            "response": gemini_text,
            "latency": gemini_lat,
            "cost": _calculate_cost("gemini-2.5-flash", gemini_usage["input_tokens"], gemini_usage["output_tokens"]),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"],
        },
    }


def compare_gemini_models(
    prompt: str,
    model_a: str = GEMINI_3_1_LITE_MODEL,
    model_b: str = GEMINI_3_5_LITE_MODEL,
) -> dict:
    """
    Compare two Gemini models (e.g. gemini-3.1-flash-lite vs gemini-3.5-flash-lite)
    with the same prompt and return a structured comparison dictionary.

    Args:
        prompt: The user message to send to both Gemini models.
        model_a: First Gemini model identifier (default: gemini-3.1-flash-lite).
        model_b: Second Gemini model identifier (default: gemini-3.5-flash-lite).

    Returns:
        A dictionary keyed by model names with response, latency, cost, and tokens.
    """
    def _calculate_cost(model_key: str, in_tokens: int, out_tokens: int) -> float:
        rates = PRICING_1M_TOKENS.get(model_key, {"input": 0.075, "output": 0.300})
        return (in_tokens * rates["input"] + out_tokens * rates["output"]) / 1_000_000.0

    resp_a, lat_a, usage_a = call_gemini(prompt, model=model_a)
    resp_b, lat_b, usage_b = call_gemini(prompt, model=model_b)

    return {
        model_a: {
            "response": resp_a,
            "latency": lat_a,
            "cost": _calculate_cost(model_a, usage_a["input_tokens"], usage_a["output_tokens"]),
            "input_tokens": usage_a["input_tokens"],
            "output_tokens": usage_a["output_tokens"],
        },
        model_b: {
            "response": resp_b,
            "latency": lat_b,
            "cost": _calculate_cost(model_b, usage_b["input_tokens"], usage_b["output_tokens"]),
            "input_tokens": usage_b["input_tokens"],
            "output_tokens": usage_b["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[types.Content] = []

    print("Chatbot started. Type 'quit' or 'exit' to end the session.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        # Keep history limited to the last 3 turns (up to 6 messages)
        current_contents = history[-6:]

        print("\nGemini: ", end="", flush=True)
        response_text = ""
        try:
            response_stream = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=current_contents,
            )
            for chunk in response_stream:
                chunk_text = getattr(chunk, "text", "") or ""
                print(chunk_text, end="", flush=True)
                response_text += chunk_text
            print()
        except Exception as e:
            print(f"\nError: {e}")

        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            )
        )
        if len(history) > 6:
            history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    if last_exception is not None:
        raise last_exception


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except TypeError:
            comparison = compare_models()
        entry = dict(comparison)
        entry["prompt"] = prompt
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare or custom comparison.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    model_mapping = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
        ("gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite"),
        ("gemini-3.5-flash-lite", "Gemini-3.5-Flash-Lite"),
        ("gemini-3.6-flash", "Gemini-3.6-Flash"),
        ("gemini-3.8-flash", "Gemini-3.8-Flash"),
    ]
    for result in results:
        prompt_str = result.get("prompt", "")
        matched_keys = set()
        for key, display_name in model_mapping:
            if key in result:
                matched_keys.add(key)
                stats = result[key]
                resp = str(stats.get("response", "")).replace("\n", " ").strip()
                if len(resp) > 50:
                    resp = resp[:47] + "..."
                latency = f"{stats.get('latency', 0.0):.2f}s"
                tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
                cost = f"${stats.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt_str} | {display_name} | {resp} | {latency} | {tokens} | {cost} |")

        # Any additional model keys
        for key, stats in result.items():
            if key != "prompt" and key not in matched_keys and isinstance(stats, dict):
                resp = str(stats.get("response", "")).replace("\n", " ").strip()
                if len(resp) > 50:
                    resp = resp[:47] + "..."
                latency = f"{stats.get('latency', 0.0):.2f}s"
                tokens = f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}"
                cost = f"${stats.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt_str} | {key} | {resp} | {latency} | {tokens} | {cost} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== SO SÁNH 2 MODEL GEMINI: gemini-3.1-flash-lite vs gemini-3.5-flash-lite ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # So sánh trực tiếp 2 phiên bản Gemini Flash Lite theo yêu cầu
        result = compare_gemini_models(test_prompt, model_a="gemini-3.1-flash-lite", model_b="gemini-3.5-flash-lite")
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response:\n{stats['response']}")

        # Hiển thị bảng Markdown tổng hợp
        table = format_comparison_table([{"prompt": test_prompt, **result}])
        print("\n=== BẢNG TỔNG HỢP SO SÁNH ===")
        print(table)
    except Exception as e:
        print(f"Skipping live Gemini comparison test: {e}")
        print("Vui lòng kiểm tra lại GEMINI_API_KEY trong file .env.")

    print("\n=== BẮT ĐẦU GEMINI 3.5 FLASH LITE CHATBOT (gõ 'quit' hoặc 'exit' để thoát) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
