import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path


class ImageGenerator:
    def __init__(self, config):
        self.config = config
        self.pipe = None

    def load(self):
        print(f"  [image] Loading model: {self.config.image_model}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.config.image_model,
            torch_dtype=torch.float32,   # fp16 produces NaN on GTX 1660 + CUDA 12.4
            safety_checker=None,
            requires_safety_checker=False,
        )

        # model_cpu_offload: only the active sub-model (UNet / VAE / text encoder)
        # lives on GPU at any moment; the rest stays in RAM.
        # Peak VRAM stays under 2 GB even in fp32, so no OOM despite other apps.
        # Note: do NOT call .to("cuda") when using cpu_offload — accelerate handles it.
        self.pipe.enable_model_cpu_offload()

        self.pipe.enable_attention_slicing(1)

        print(f"  [image] VRAM after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("  [image] Model unloaded, VRAM freed")

    def generate(self, prompt: str, output_path: Path) -> Path:
        if self.config.skip_existing and output_path.exists():
            return output_path

        full_prompt = f"{self.config.image_style_prefix}, {prompt}"

        result = self.pipe(
            full_prompt,
            negative_prompt=self.config.image_negative_prompt,
            num_inference_steps=self.config.image_steps,
            guidance_scale=self.config.image_guidance_scale,
            width=self.config.image_width,
            height=self.config.image_height,
        )
        result.images[0].save(str(output_path))
        return output_path
