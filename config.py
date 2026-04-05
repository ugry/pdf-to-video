from dataclasses import dataclass


@dataclass
class Config:
    # Input
    pdf_path: str = ""

    # Output
    output_dir: str = "output"

    # LLM (Ollama) - used to convert page text into an image prompt
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    # Image generation (Stable Diffusion via diffusers)
    image_model: str = "Lykon/dreamshaper-8"
    image_style_prefix: str = "cartoon illustration, storybook art style,"
    image_negative_prompt: str = (
        "ugly, blurry, deformed, bad anatomy, nsfw, text, watermark, signature"
    )
    image_width: int = 512
    image_height: int = 512
    image_steps: int = 25
    image_guidance_scale: float = 7.5

    # TTS (Kokoro)
    tts_voice: str = "af_heart"   # American English female, warm
    tts_speed: float = 1.0
    tts_lang: str = "a"           # 'a' = American English

    # Video
    fps: int = 24
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    ken_burns: bool = True         # subtle zoom-in effect on images

    # Processing
    max_words_per_page: int = 600  # truncate dense pages for TTS
    min_words_per_page: int = 10   # skip near-blank pages
    skip_existing: bool = True     # resume interrupted runs
