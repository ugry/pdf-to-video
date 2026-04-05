#!/usr/bin/env python3
"""
pdf_to_video — turns a PDF book into a narrated cartoon video.

Usage:
    python main.py book.pdf
    python main.py book.pdf --output my_output --voice af_heart --steps 30
    python main.py book.pdf --pages 1-10        # process only pages 1–10
    python main.py book.pdf --no-ken-burns
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from config import Config
from pipeline.pdf_extractor import extract_pages
from pipeline.prompt_generator import generate_prompt
from pipeline.image_generator import ImageGenerator
from pipeline.tts_engine import TTSEngine
from pipeline.video_assembler import VideoAssembler


# ── logging setup ─────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    logger = logging.getLogger("pdf_to_video")
    logger.setLevel(logging.DEBUG)

    # file handler — full debug output
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    # console handler — info and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_page_range(s: str):
    if not s:
        return None
    parts = s.split("-")
    return int(parts[0]), int(parts[1])


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def is_valid_image(path: Path) -> bool:
    """Returns False if the image is missing, empty, or corrupt."""
    if not path.exists():
        return False
    if path.stat().st_size < 1024:   # anything under 1 KB is almost certainly empty
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def is_valid_audio(path: Path) -> bool:
    """Returns False if the audio file is missing or suspiciously small."""
    if not path.exists():
        return False
    if path.stat().st_size < 512:
        return False
    return True


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(config: Config, page_range, log: logging.Logger):
    pdf_path = Path(config.pdf_path)
    if not pdf_path.exists():
        log.error(f"PDF not found: {pdf_path}")
        sys.exit(1)

    output_dir = Path(config.output_dir)
    img_dir = output_dir / "images"
    aud_dir = output_dir / "audio"
    img_dir.mkdir(parents=True, exist_ok=True)
    aud_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'='*60}")
    log.info(f"PDF:    {pdf_path}")
    log.info(f"Model:  {config.image_model}")
    log.info(f"Voice:  {config.tts_voice}")
    log.info(f"Output: {output_dir}")
    log.info(f"{'='*60}")

    # ── 1. Extract text ───────────────────────────────────────────────────────
    log.info(f"\n[1/4] Extracting pages from {pdf_path.name}...")
    all_pages = extract_pages(str(pdf_path))

    if page_range:
        lo, hi = page_range
        all_pages = [p for p in all_pages if lo <= p["page_num"] <= hi]

    pages = [p for p in all_pages if len(p["text"].split()) >= config.min_words_per_page]
    skipped = len(all_pages) - len(pages)
    log.info(f"  {len(pages)} pages to process  ({skipped} blank/short skipped)")

    if not pages:
        log.error("No processable pages found. Exiting.")
        sys.exit(0)

    for p in pages:
        log.debug(f"  page {p['page_num']}: {len(p['text'].split())} words")

    # ── 2. Generate images ────────────────────────────────────────────────────
    log.info(f"\n[2/4] Generating cartoon images ({config.image_width}×{config.image_height})...")
    image_gen = ImageGenerator(config)
    image_gen.load()
    failed_images = []

    for i, page in enumerate(pages, 1):
        img_path = img_dir / f"page_{page['page_num']:04d}.png"

        if config.skip_existing and is_valid_image(img_path):
            log.info(f"  [{i}/{len(pages)}] page {page['page_num']} — image exists, skipping")
            page["image"] = img_path
            continue

        # delete invalid/empty leftover if present
        if img_path.exists():
            log.warning(f"  [{i}/{len(pages)}] page {page['page_num']} — existing image is invalid, regenerating")
            img_path.unlink()

        try:
            log.info(f"  [{i}/{len(pages)}] page {page['page_num']} — generating prompt...")
            prompt = generate_prompt(page["text"], config)
            log.info(f"    prompt: {prompt[:100]}")
            log.debug(f"    full prompt: {prompt}")

            log.info(f"    generating image...")
            image_gen.generate(prompt, img_path)

            if not is_valid_image(img_path):
                raise RuntimeError(f"Generated image is empty or corrupt: {img_path}")

            size_kb = img_path.stat().st_size // 1024
            log.info(f"    saved ({size_kb} KB): {img_path.name}")
            page["image"] = img_path

        except Exception as e:
            log.error(f"  [{i}/{len(pages)}] page {page['page_num']} — IMAGE FAILED: {e}")
            log.debug(traceback.format_exc())
            failed_images.append(page["page_num"])
            # continue with remaining pages

    image_gen.unload()

    if failed_images:
        log.warning(f"  {len(failed_images)} image(s) failed: pages {failed_images}")

    # ── 3. Generate audio ─────────────────────────────────────────────────────
    log.info(f"\n[3/4] Generating TTS audio...")
    tts = TTSEngine(config)
    tts.load()
    failed_audio = []

    for i, page in enumerate(pages, 1):
        aud_path = aud_dir / f"page_{page['page_num']:04d}.wav"

        if config.skip_existing and is_valid_audio(aud_path):
            log.info(f"  [{i}/{len(pages)}] page {page['page_num']} — audio exists, skipping")
            page["audio"] = aud_path
            continue

        if aud_path.exists():
            log.warning(f"  [{i}/{len(pages)}] page {page['page_num']} — existing audio is invalid, regenerating")
            aud_path.unlink()

        try:
            log.info(f"  [{i}/{len(pages)}] page {page['page_num']} — synthesizing speech...")
            text = truncate_words(page["text"], config.max_words_per_page)
            log.debug(f"    text ({len(text.split())} words): {text[:80]}...")
            tts.generate(text, aud_path)

            if not is_valid_audio(aud_path):
                raise RuntimeError(f"Generated audio is empty: {aud_path}")

            size_kb = aud_path.stat().st_size // 1024
            log.info(f"    saved ({size_kb} KB): {aud_path.name}")
            page["audio"] = aud_path

        except Exception as e:
            log.error(f"  [{i}/{len(pages)}] page {page['page_num']} — AUDIO FAILED: {e}")
            log.debug(traceback.format_exc())
            failed_audio.append(page["page_num"])

    tts.unload()

    if failed_audio:
        log.warning(f"  {len(failed_audio)} audio(s) failed: pages {failed_audio}")

    # ── 4. Assemble video ─────────────────────────────────────────────────────
    log.info(f"\n[4/4] Assembling final video...")
    video_path = output_dir / f"{pdf_path.stem}.mp4"

    page_data = [
        {"image": p["image"], "audio": p["audio"], "page_num": p["page_num"]}
        for p in pages
        if "image" in p and "audio" in p
        and is_valid_image(p["image"]) and is_valid_audio(p["audio"])
    ]

    if not page_data:
        log.error("No valid image+audio pairs to assemble. Check the log for errors.")
        sys.exit(1)

    skipped_assembly = len(pages) - len(page_data)
    if skipped_assembly:
        log.warning(f"  {skipped_assembly} page(s) skipped in assembly due to missing image or audio")

    try:
        assembler = VideoAssembler(config)
        assembler.assemble(page_data, video_path)
        log.info(f"\nDone! Video: {video_path}  ({video_path.stat().st_size // 1024 // 1024} MB)")
    except Exception as e:
        log.error(f"Video assembly failed: {e}")
        log.debug(traceback.format_exc())
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert a PDF book into a narrated cartoon video.")
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--pages", default="", help="Page range e.g. 1-20")
    parser.add_argument("--voice", default="af_heart", help="Kokoro TTS voice")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion steps")
    parser.add_argument("--model", default="Lykon/dreamshaper-8", help="HuggingFace image model ID")
    parser.add_argument("--ollama-model", default="llama3.2:3b", help="Ollama model for prompts")
    parser.add_argument("--no-ken-burns", action="store_true", help="Disable Ken Burns zoom effect")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all files")
    args = parser.parse_args()

    config = Config(
        pdf_path=args.pdf,
        output_dir=args.output,
        tts_voice=args.voice,
        tts_speed=args.speed,
        image_steps=args.steps,
        image_model=args.model,
        ollama_model=args.ollama_model,
        ken_burns=not args.no_ken_burns,
        skip_existing=not args.no_resume,
    )

    log = setup_logging(Path(args.output))
    log.info(f"Log file: {Path(args.output) / 'run.log'}")

    page_range = parse_page_range(args.pages)
    run(config, page_range, log)


if __name__ == "__main__":
    main()
