import ollama


SYSTEM_PROMPT = """You are a visual prompt engineer for a cartoon storybook app.
Your job: read a passage of text and return a single image generation prompt
that captures the most visually interesting scene from it.

Rules:
- Maximum 40 words
- Describe scene, characters, setting, and mood concisely
- Do NOT include any explanation — output the prompt only
- Always end with: cartoon style, colorful, storybook illustration"""

FALLBACK_TEMPLATE = (
    "A scene from a story, {summary}, cartoon style, colorful, storybook illustration"
)


def generate_prompt(text: str, config) -> str:
    """
    Calls Ollama to turn page text into an image generation prompt.
    Falls back to a simple truncated summary if Ollama is unavailable.
    """
    try:
        response = ollama.chat(
            model=config.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:1500]},
            ],
            options={"temperature": 0.7},
            keep_alive=0,
        )
        prompt = response["message"]["content"].strip()
        # safety: strip markdown fences if model wrapped the output
        prompt = prompt.strip("`").strip()
        return prompt
    except Exception as e:
        print(f"  [prompt] Ollama unavailable ({e}), using fallback prompt")
        summary = " ".join(text.split()[:15])
        return FALLBACK_TEMPLATE.format(summary=summary)
