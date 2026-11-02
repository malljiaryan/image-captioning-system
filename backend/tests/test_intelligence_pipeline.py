import io
import os
import sys
from PIL import Image

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.ai.semantic_analyzer import analyze_image_semantics
from app.ai.evaluator import (
    calculate_bleu,
    calculate_rouge,
    calculate_meteor,
    calculate_semantic_similarity,
    evaluate_caption
)

def get_test_image():
    buf = io.BytesIO()
    # Image of blue sky and grass
    img = Image.new("RGB", (320, 240), color=(100, 180, 240))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def run_intelligence_tests():
    print("==========================================================")
    print("  Milestone 3.9: Advanced AI Caption Intelligence Tests   ")
    print("==========================================================")

    # 1. Semantic Scene Understanding Test
    print("\n[Test 1] Semantic Scene Analysis")
    doc_path = "uploads/New_Doc_06-24-2026_17.17_page-0001.jpg"
    test_target = doc_path if os.path.exists(doc_path) else get_test_image()
    semantics = analyze_image_semantics(test_target)
    print(f"  -> Detected Objects   : {semantics['objects']}")
    print(f"  -> Scene Context       : {semantics['scene']}")
    print(f"  -> Activity / State    : {semantics['activity']}")
    print(f"  -> Visual Attributes   : {semantics['attributes']}")
    print(f"  -> Context Summary     : {semantics['context_summary']}")
    assert isinstance(semantics["objects"], list) and len(semantics["objects"]) > 0
    assert isinstance(semantics["scene"], str) and len(semantics["scene"]) > 0

    # 2. BLEU Metric Test
    print("\n[Test 2] BLEU-1..4 Evaluation")
    cand = "a dog playing with a ball in the garden"
    ref = "a dog playing with a small red ball in the green garden"
    bleu = calculate_bleu(cand, ref)
    print(f"  -> BLEU-1: {bleu['bleu_1']}%, BLEU-2: {bleu['bleu_2']}%, BLEU-4: {bleu['bleu_4']}%")
    print(f"  -> Cumulative BLEU: {bleu['cumulative_bleu']}%, BP: {bleu['brevity_penalty']}")
    assert bleu["bleu_1"] > 0
    assert 0.0 <= bleu["cumulative_bleu"] <= 100.0

    # 3. ROUGE Metric Test
    print("\n[Test 3] ROUGE-1/2/L Evaluation")
    rouge = calculate_rouge(cand, ref)
    print(f"  -> ROUGE-1 F1: {rouge['rouge_1']['f1']}%")
    print(f"  -> ROUGE-2 F1: {rouge['rouge_2']['f1']}%")
    print(f"  -> ROUGE-L F1: {rouge['rouge_l']['f1']}%")
    assert rouge["rouge_1"]["f1"] > 0
    assert rouge["rouge_l"]["f1"] > 0

    # 4. METEOR Metric Test
    print("\n[Test 4] METEOR Evaluation")
    meteor = calculate_meteor(cand, ref)
    print(f"  -> METEOR Score: {meteor['meteor']}% (F_mean: {meteor['f_mean']}%, penalty: {meteor['penalty']})")
    assert meteor["meteor"] > 0

    # 5. Semantic Similarity Test
    print("\n[Test 5] Semantic Similarity")
    sim = calculate_semantic_similarity(cand, ref)
    print(f"  -> Similarity: {sim['similarity_score']}% ({sim['level']})")
    print(f"  -> Shared Tokens: {sim['shared_tokens']}")
    assert sim["similarity_score"] > 50.0

    # 6. Comprehensive Caption Evaluation
    print("\n[Test 6] Full Caption Evaluation Pipeline")
    eval_ref = evaluate_caption(cand, reference=ref, image_semantics=semantics)
    print(f"  -> Composite Score : {eval_ref['composite_score']}/100")
    print(f"  -> Quality Grade   : {eval_ref['grade']}")
    assert eval_ref["composite_score"] > 50.0
    assert eval_ref["grade"] in ["Exceptional", "High Quality", "Good"]

    # 7. Flask /api/evaluate Endpoint Test
    print("\n[Test 7] Flask API POST /api/evaluate")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    api_eval = client.post("/api/evaluate", json={
        "candidate": cand,
        "reference": ref
    })
    assert api_eval.status_code == 200
    eval_resp = api_eval.get_json()
    assert eval_resp["success"] is True
    assert "evaluation" in eval_resp
    assert eval_resp["evaluation"]["bleu"]["bleu_1"] > 0
    print(f"  -> API Evaluation Response OK: Composite={eval_resp['evaluation']['composite_score']}")

    # 8. Flask /api/caption Endpoint (Semantics & Quality Verification)
    print("\n[Test 8] Flask API POST /api/caption with Semantics & Quality")
    caption_res = client.post("/api/caption", data={
        "image": (get_test_image(), "pipeline_test.jpg"),
        "style": "standard"
    }, content_type="multipart/form-data")
    assert caption_res.status_code == 200
    caption_json = caption_res.get_json()
    assert caption_json["success"] is True
    assert "semantics" in caption_json and len(caption_json["semantics"]["objects"]) > 0
    assert "quality_evaluation" in caption_json and caption_json["quality_evaluation"]["composite_score"] > 0
    print(f"  -> Captions with Semantics OK: Objects={caption_json['semantics']['objects']}, Score={caption_json['quality_evaluation']['composite_score']}")

    print("\n[SUCCESS] All Milestone 3.9 intelligence pipeline tests passed successfully!")


if __name__ == "__main__":
    run_intelligence_tests()
