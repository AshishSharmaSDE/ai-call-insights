import requests
from app.core.config import settings

# try to import ollama, but allow fallback
try:
    import ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False


def analyze_sentiment(text: str) -> str:
    """
    Returns one of: Positive, Negative, Neutral (simple prompt-based)
    Switches between external LLM API and local Ollama (if available).
    """
    prompt = f"Analyze the sentiment of this text: '{text}'. Respond only with Positive, Negative, or Neutral."

    if settings.USE_LLAMA_API and settings.LLAMA_API_KEY and settings.LLAMA_API_BASE:
        headers = {"Authorization": f"Bearer {settings.LLAMA_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": settings.LLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }
        resp = requests.post(settings.LLAMA_API_BASE, headers=headers, json=payload)
        resp.raise_for_status()
        choices = resp.json().get("choices")
        if choices and len(choices) > 0:
            return choices[0]["message"]["content"].strip()
        return resp.json().get("text", "").strip()

    # Local Ollama if installed
    if _OLLAMA_AVAILABLE:
        resp = ollama.chat(model=settings.LLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return resp.get("message", {}).get("content", "").strip()

    # Fallback: a very small heuristic (fast) if no LLM is available
    low = text.lower()
    if any(w in low for w in ["not happy", "angry", "upset", "frustrat"]):
        return "Negative"
    if any(w in low for w in ["thank", "great", "happy", "good", "satisfied"]):
        return "Positive"
    return "Neutral"