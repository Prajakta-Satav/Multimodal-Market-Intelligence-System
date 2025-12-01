# app/core/llm.py
from app.core.config import settings
from app.core.logging import logger

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def call_llm(prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
    provider = getattr(settings, "LLM_PROVIDER", "gemini").lower()
    logger.info(f"[LLM] Using provider={provider}")

    if provider == "gemini" and genai:
        genai.configure(api_key=settings.GEMINI_API_KEYS[0])
        model = genai.GenerativeModel(settings.GEMINI_MODEL or "gemini-1.5-pro")

        def _generate(p: str, temp: float):
            return model.generate_content(
                p,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                },
            )

        # First attempt
        response = _generate(prompt, temperature)
        candidate = response.candidates[0] if response.candidates else None

        if not candidate or not candidate.content.parts:
            logger.warning(
                f"[LLM] No parts returned (finish_reason={getattr(candidate,'finish_reason',None)}). Retrying..."
            )
            # Retry with a slightly stronger instruction and lower temperature
            retry_prompt = prompt + "\n\nPlease provide at least one clear sentence, even if context is limited."
            response = _generate(retry_prompt, 0.2)
            candidate = response.candidates[0] if response.candidates else None

            if not candidate or not candidate.content.parts:
                logger.error("[LLM] Retry also failed, returning sentinel")
                return "(no answer generated)"

        # Collect text parts
        texts = [p.text for p in candidate.content.parts if hasattr(p, "text")]
        llm_output = "".join(texts).strip()

        if not llm_output:
            logger.warning("[LLM] Gemini returned empty text output after retry")
            return "(no answer generated)"

        return llm_output

    else:
        raise RuntimeError(f"Unsupported or unavailable LLM provider: {provider}")
