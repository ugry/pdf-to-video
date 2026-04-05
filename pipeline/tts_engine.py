import numpy as np
import soundfile as sf
from pathlib import Path


class TTSEngine:
    def __init__(self, config):
        self.config = config
        self._pipeline = None

    def load(self):
        from kokoro import KPipeline
        print(f"  [tts] Loading Kokoro (lang={self.config.tts_lang})")
        self._pipeline = KPipeline(lang_code=self.config.tts_lang, device='cpu')

    def unload(self):
        self._pipeline = None
        import torch
        torch.cuda.empty_cache()
        print("  [tts] Engine unloaded")

    def generate(self, text: str, output_path: Path) -> Path:
        if self.config.skip_existing and output_path.exists():
            return output_path

        chunks = []
        generator = self._pipeline(
            text,
            voice=self.config.tts_voice,
            speed=self.config.tts_speed,
        )
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                chunks.append(audio)

        if chunks:
            audio_data = np.concatenate(chunks)
        else:
            # silence placeholder for pages that produce no audio
            audio_data = np.zeros(24000, dtype=np.float32)

        sf.write(str(output_path), audio_data, 24000)
        return output_path
