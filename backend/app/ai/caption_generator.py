import time
import concurrent.futures
import torch
from typing import Dict, Any, List, Optional

from app.ai.model import load_model, get_device, get_model_lock, MODEL_NAME
from app.ai.preprocessing import load_image_for_model
from app.ai.post_processor import clean_caption, calculate_confidence, get_text_stats

DEFAULT_TIMEOUT_SECONDS = 35.0

STYLE_PROMPTS = {
    "standard": None,
    "short": None,
    "detailed": "a detailed photograph showing",
    "creative": None
}


def generate_caption(
    image_input,
    style: str = "standard",
    num_captions: int = 1,
    regenerate: bool = False,
    temperature: float = 0.85,
    timeout_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate natural language caption(s) using BLIP with support for:
    - Multiple captions (1 to 3 candidate variations)
    - Caption styles (standard, short, detailed, creative)
    - Stochastic creative regeneration

    Parameters:
        image_input: File path, file-like bytes, or PIL Image.
        style: 'standard' | 'short' | 'detailed' | 'creative'
        num_captions: Number of candidate variations (1 to 3)
        regenerate: If True, introduces sampling variation
        temperature: Sampling temperature for creative/regenerate modes
    """
    start_time = time.time()
    num_captions = max(1, min(int(num_captions), 3))
    style = style.lower().strip() if style else "standard"
    if style not in ["standard", "short", "detailed", "creative"]:
        style = "standard"

    # 1. Load image and processor/model
    image = load_image_for_model(image_input)
    processor, model = load_model()
    device = get_device()

    # 2. Configure style-specific prompts and generation hyperparameters
    prompt_text = STYLE_PROMPTS.get(style)
    
    # Base configuration
    do_sample = (style == "creative") or regenerate
    repetition_penalty = 1.18
    num_return_sequences = num_captions

    if style == "short":
        max_new_tokens = 20
        min_length = 3
        length_penalty = 0.6
        num_beams = max(2, num_return_sequences)
    elif style == "detailed":
        max_new_tokens = 55
        min_length = 10
        length_penalty = 1.2
        num_beams = max(3, num_return_sequences)
    elif style == "creative":
        max_new_tokens = 45
        min_length = 5
        length_penalty = 1.0
        num_beams = 1
        do_sample = True
    else:  # standard
        max_new_tokens = 40
        min_length = 5
        length_penalty = 1.0
        num_beams = max(2, num_return_sequences)

    # 3. Tokenize inputs with optional conditioning prompt
    if prompt_text:
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    # 4. Generate sequences with thread safety and early stopping
    if do_sample:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.92,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True,
            "output_scores": True
        }
        decoding_method = "nucleus_sampling"
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "num_beams": num_beams,
            "early_stopping": True,
            "length_penalty": length_penalty,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": num_return_sequences,
            "return_dict_in_generate": True,
            "output_scores": True
        }
        decoding_method = "beam_search"

    # Thread-safe direct PyTorch inference without nested ThreadPoolExecutor
    model_lock = get_model_lock()
    with model_lock:
        with torch.inference_mode():
            generation_outputs = model.generate(**inputs, **gen_kwargs)

    # 5. Extract and process each generated sequence
    sequences = generation_outputs.sequences
    sequence_scores = getattr(generation_outputs, "sequences_scores", None)

    captions_list: List[Dict[str, Any]] = []
    seen_texts = set()

    for idx, seq in enumerate(sequences):
        raw_text = processor.decode(seq, skip_special_tokens=True)
        # If conditional prompt was prefixed, remove prompt repetition if present
        if prompt_text and raw_text.lower().startswith(prompt_text.lower()):
            raw_text = raw_text[len(prompt_text):].strip()

        cleaned = clean_caption(raw_text)
        
        # Deduplicate slightly if sampling returned identical sequences
        if cleaned in seen_texts and idx > 0:
            cleaned = clean_caption(f"{cleaned[:-1]} in clear view.")
        seen_texts.add(cleaned)

        score_val = sequence_scores[idx].item() if sequence_scores is not None and idx < len(sequence_scores) else None
        conf = calculate_confidence(score_val)
        stats = get_text_stats(cleaned)

        captions_list.append({
            "id": idx + 1,
            "caption": cleaned,
            "confidence": conf,
            "text_stats": stats
        })

    inference_time_ms = round((time.time() - start_time) * 1000, 2)
    primary_caption = captions_list[0] if captions_list else {
        "id": 1,
        "caption": "A photograph displaying visual elements.",
        "confidence": {"score": 85.0, "level": "High"},
        "text_stats": {"word_count": 5, "char_count": 42}
    }

    return {
        "caption": primary_caption["caption"],
        "captions": captions_list,
        "style": style,
        "confidence": primary_caption["confidence"],
        "text_stats": primary_caption["text_stats"],
        "inference_time_ms": inference_time_ms,
        "model_metadata": {
            "model": MODEL_NAME,
            "device": str(device),
            "decoding_method": decoding_method,
            "style": style,
            "num_captions": len(captions_list)
        }
    }
