import re
import torch
from typing import Dict, Any, List

from app.ai.model import load_model, get_device, get_model_lock
from app.ai.preprocessing import load_image_for_model
from app.ai.post_processor import clean_caption


def parse_entities(text: str, remove_prefix: str = "") -> List[str]:
    """Parse comma/conjunction-separated entity names from probed text."""
    if not text:
        return []
    if remove_prefix and text.lower().startswith(remove_prefix.lower()):
        text = text[len(remove_prefix):].strip()

    # Clean punctuation and common boilerplate
    text = re.sub(r"^(is|are|shows|contains|features)\s+", "", text, flags=re.IGNORECASE)
    text = text.replace(" and ", ", ").replace(" with ", ", ")

    items = [re.sub(r"[^\w\s-]", "", item).strip() for item in text.split(",")]
    # Deduplicate while preserving order, filter out short noise
    unique_items = []
    seen = set()
    for item in items:
        cleaned_item = item.strip().lower()
        if cleaned_item and len(cleaned_item) > 2 and cleaned_item not in seen:
            seen.add(cleaned_item)
            unique_items.append(item.strip().capitalize())
            if len(unique_items) >= 4:
                break

    return unique_items


def analyze_image_semantics(image_input) -> Dict[str, Any]:
    """
    Perform deep visual semantic analysis on an image using BLIP vision-language probes.
    Extracts:
        - Major objects
        - Scene/environment context
        - Action/activity taking place
        - Visual attributes and colors
        - Synthesized context-aware summary
    """
    image = load_image_for_model(image_input)
    processor, model = load_model()
    device = get_device()

    # Optimized consolidated probing prompts (reduces forward passes from 4 to 2)
    probes = {
        "objects": "the main objects and setting in this image are",
        "activity": "the action and visual colors in this image are"
    }

    raw_results = {}
    model_lock = get_model_lock()

    with model_lock:
        with torch.inference_mode():
            for key, prompt in probes.items():
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                output = model.generate(
                    **inputs,
                    max_new_tokens=18,
                    num_beams=1,
                    no_repeat_ngram_size=2
                )
                decoded = processor.decode(output[0], skip_special_tokens=True)
                raw_results[key] = decoded

    # Parse and structure probed concepts
    objects_list = parse_entities(raw_results.get("objects", ""), probes["objects"])
    if not objects_list:
        objects_list = ["Visual subject", "Foreground element"]

    raw_scene = raw_results.get("objects", "")
    scene_context = clean_caption(raw_scene, fallback="Visual environment.").rstrip(".")

    raw_activity = raw_results.get("activity", "")
    activity_desc = clean_caption(raw_activity, fallback="Static scene.").rstrip(".")

    attributes_list = parse_entities(raw_results.get("activity", ""), probes["activity"])
    if not attributes_list:
        attributes_list = ["Natural lighting", "Clear contrast"]

    # Synthesize unified context-aware summary
    context_summary = f"A scene depicting {scene_context.lower()}, highlighting {', '.join(objects_list[:2]).lower()}."

    return {
        "objects": objects_list,
        "scene": scene_context,
        "activity": activity_desc,
        "attributes": attributes_list,
        "context_summary": context_summary
    }
