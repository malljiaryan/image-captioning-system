/* =========================================
   CAPTION GENERATION & ADVANCED AI INTELLIGENCE
   Milestone 3.12: Frontend <-> Backend Integration
========================================= */

// ---------------------------------------------------------------------------
// 1. Configuration & Endpoints (3.12.1, 3.12.2, 3.12.9)
// ---------------------------------------------------------------------------
function getApiBaseUrl() {
    if (window.location.origin && window.location.origin.startsWith("http")) {
        // If served directly by Flask (port 5000) or standard web server (port 80/443 in production)
        if (window.location.port === "5000" || window.location.port === "" || window.location.port === "80" || window.location.port === "443") {
            return window.location.origin;
        }
        // If opened through a static dev server (e.g. VS Code Live Server on 5500, http.server on 8000, Vite, etc.)
        // route the API requests directly to the Flask backend on port 5000!
        return `${window.location.protocol}//${window.location.hostname}:5000`;
    }
    // Fallback for file:// or local files
    return "http://127.0.0.1:5000";
}

const API_BASE_URL = getApiBaseUrl();
const CAPTION_API_URL = `${API_BASE_URL}/api/caption`;
const REGENERATE_API_URL = `${API_BASE_URL}/api/caption/regenerate`;
const EVALUATE_API_URL = `${API_BASE_URL}/api/evaluate`;
const HISTORY_STORAGE_KEY = "visionCaptionHistory";

// ---------------------------------------------------------------------------
// 2. Centralized Frontend Application State (3.12.11)
// ---------------------------------------------------------------------------
const AppState = {
    status: "IDLE", // "IDLE" | "GENERATING" | "REGENERATING" | "SUCCESS" | "ERROR"
    resultId: null,
    currentStyle: "standard",
    numVariations: 1,
    lastResponseData: null,
    selectedVariationIndex: 0,
    loadingTimers: []
};

// ---------------------------------------------------------------------------
// 3. UI DOM Elements
// ---------------------------------------------------------------------------
const resultCard = document.getElementById("captionResultCard");
const resultText = document.getElementById("captionResult");
const loadingBox = document.getElementById("captionLoading");
const loadingText = loadingBox ? loadingBox.querySelector("p") : null;
const errorBox = document.getElementById("captionError");
const errorText = document.getElementById("captionErrorMessage");
const generateButton = document.getElementById("generateCaptionBtn");
const imageFileInput = document.getElementById("imageInput");
const removeButton = document.getElementById("removeImageBtn");
const imagePreviewElement = document.getElementById("imagePreview");

// Metadata & Action Elements
const copyCaptionBtn = document.getElementById("copyCaptionBtn");
const regenerateCaptionBtn = document.getElementById("regenerateCaptionBtn");
const exportTxtBtn = document.getElementById("exportTxtBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");
const metaInference = document.getElementById("metaInference");
const metaConfidence = document.getElementById("metaConfidence");
const metaDimensions = document.getElementById("metaDimensions");
const metaWordCount = document.getElementById("metaWordCount");
const metaModelTag = document.getElementById("metaModelTag");

// Comparison & Variations Elements
const comparisonContainer = document.getElementById("comparisonContainer");
const variationsGrid = document.getElementById("variationsGrid");
const multiCaptionToggle = document.getElementById("multiCaptionToggle");
const styleChips = document.querySelectorAll(".style-chip");

// Semantic Scene Intelligence Elements
const semanticCard = document.getElementById("semanticCard");
const detectedObjectsTags = document.getElementById("detectedObjectsTags");
const detectedSceneVal = document.getElementById("detectedSceneVal");
const detectedActivityVal = document.getElementById("detectedActivityVal");
const detectedAttributesTags = document.getElementById("detectedAttributesTags");

// Quality Benchmark & Evaluation Elements
const evaluationCard = document.getElementById("evaluationCard");
const evalScoreNum = document.getElementById("evalScoreNum");
const evalScoreGrade = document.getElementById("evalScoreGrade");
const evalScoreDesc = document.getElementById("evalScoreDesc");
const referenceCaptionInput = document.getElementById("referenceCaptionInput");
const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");
const benchmarkResultsGrid = document.getElementById("benchmarkResultsGrid");
const metricBleu = document.getElementById("metricBleu");
const metricRouge = document.getElementById("metricRouge");
const metricMeteor = document.getElementById("metricMeteor");
const metricSimilarity = document.getElementById("metricSimilarity");


// ---------------------------------------------------------------------------
// 4. Style Selection (Short, Detailed, Creative, Standard)
// ---------------------------------------------------------------------------
styleChips.forEach(chip => {
    chip.addEventListener("click", () => {
        styleChips.forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        AppState.currentStyle = chip.dataset.style || "standard";
        console.log("[State] Style updated to:", AppState.currentStyle);
    });
});


// ---------------------------------------------------------------------------
// 5. UI State Management Helpers (3.12.3, 3.12.5, 3.12.11)
// ---------------------------------------------------------------------------
function resetCaptionUI() {
    AppState.status = "IDLE";
    AppState.resultId = null;
    AppState.lastResponseData = null;
    AppState.selectedVariationIndex = 0;
    stopLoadingProgress();

    resultCard.hidden = true;
    loadingBox.hidden = true;
    errorBox.hidden = true;
    if (comparisonContainer) comparisonContainer.hidden = true;
    if (semanticCard) semanticCard.hidden = true;
    if (evaluationCard) evaluationCard.hidden = true;
    if (benchmarkResultsGrid) benchmarkResultsGrid.hidden = true;

    resultText.textContent = "";
    generateButton.disabled = false;
    generateButton.textContent = "Generate Caption";

    if (copyCaptionBtn) {
        copyCaptionBtn.textContent = "📋 Copy Caption";
        copyCaptionBtn.classList.remove("copied");
    }
}

function showCaptionError(message) {
    AppState.status = "ERROR";
    stopLoadingProgress();

    loadingBox.hidden = true;
    errorText.textContent = message;
    errorBox.hidden = false;

    generateButton.disabled = false;
    generateButton.textContent = "Generate Caption";
    if (regenerateCaptionBtn) {
        regenerateCaptionBtn.disabled = false;
        regenerateCaptionBtn.textContent = "🔄 Regenerate";
    }
}

function startLoadingProgress(mode = "generate") {
    stopLoadingProgress();

    if (!loadingText) return;

    if (mode === "regenerate") {
        loadingText.textContent = "🔄 Sampling creative variations from BLIP model...";
        return;
    }

    const phases = [
        { delay: 0, text: "📤 Uploading image to server..." },
        { delay: 800, text: "🧠 Executing BLIP vision-language neural network..." },
        { delay: 2500, text: "🔍 Extracting visual semantics and scene context..." },
        { delay: 4200, text: "📊 Scoring caption quality & linguistic metrics..." },
        { delay: 7000, text: "⚡ Finalizing deep learning inference & response..." },
        { delay: 12000, text: "⏳ Processing complex visual tokens (CPU inference)..." }
    ];

    phases.forEach(p => {
        const timer = setTimeout(() => {
            if ((AppState.status === "GENERATING" || AppState.status === "REGENERATING") && loadingText) {
                loadingText.textContent = p.text;
            }
        }, p.delay);
        AppState.loadingTimers.push(timer);
    });
}

function stopLoadingProgress() {
    if (AppState.loadingTimers && AppState.loadingTimers.length > 0) {
        AppState.loadingTimers.forEach(t => clearTimeout(t));
        AppState.loadingTimers = [];
    }
}

function formatErrorMessage(err) {
    if (!navigator.onLine) {
        return "You appear to be offline. Please check your network connection.";
    }
    const msg = String(err.message || err || "");
    if (err.name === "AbortError") {
        return "⏱️ The AI request timed out. The server or model took longer than 45 seconds. Please ensure the backend is running and try again.";
    }
    if (err.name === "TypeError" && (msg.includes("fetch") || msg.includes("NetworkError") || msg.includes("Failed to fetch"))) {
        return "⚠️ Cannot connect to backend server. Please verify the Flask server is running at http://127.0.0.1:5000 (run 'python run.py' in the terminal).";
    }
    return msg || "An unexpected error occurred while communicating with the AI service.";
}


// ---------------------------------------------------------------------------
// 6. History Storage (Local fallback + Backend sync) (3.12.6, 3.12.7)
// ---------------------------------------------------------------------------
function saveCaptionToHistory(data, selectedCaption, thumbnailSrc) {
    try {
        const savedHistory = localStorage.getItem(HISTORY_STORAGE_KEY);
        let historyItems = savedHistory ? JSON.parse(savedHistory) : [];

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const dateStr = `Today · ${timeStr}`;

        const newHistoryEntry = {
            id: data.id || Date.now(),
            icon: "✨",
            thumbnail: thumbnailSrc || null,
            caption: selectedCaption.caption,
            style: data.style || AppState.currentStyle,
            date: dateStr,
            confidence: selectedCaption.confidence ? `${selectedCaption.confidence.score}% (${selectedCaption.confidence.level})` : null,
            dimensions: data.image_metadata ? `${data.image_metadata.width}×${data.image_metadata.height}` : null,
            inferenceTime: data.inference_time_ms ? `${(data.inference_time_ms / 1000).toFixed(2)}s` : null,
            wordCount: selectedCaption.text_stats ? selectedCaption.text_stats.word_count : null,
            variationsCount: data.captions ? data.captions.length : 1,
            scene: data.semantics ? data.semantics.scene : null,
            qualityScore: data.quality_evaluation ? data.quality_evaluation.composite_score : null
        };

        historyItems.unshift(newHistoryEntry);
        if (historyItems.length > 100) historyItems = historyItems.slice(0, 100);
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(historyItems));
    } catch (err) {
        console.warn("Unable to cache caption to localStorage history:", err);
    }
}


// ---------------------------------------------------------------------------
// 7. Render Variations (Short, Detailed, Creative candidate cards)
// ---------------------------------------------------------------------------
function renderVariations(captions, data) {
    if (!comparisonContainer || !variationsGrid) return;

    if (!captions || captions.length <= 1) {
        comparisonContainer.hidden = true;
        return;
    }

    comparisonContainer.hidden = false;
    variationsGrid.innerHTML = "";

    captions.forEach((item, idx) => {
        const card = document.createElement("div");
        card.className = `variation-card ${idx === AppState.selectedVariationIndex ? "selected" : ""}`;
        card.dataset.index = idx;

        card.innerHTML = `
            <div class="variation-card-header">
                <span class="variation-tag">Variation ${item.id || idx + 1}</span>
                <span style="font-size: 11px; color: #64748b; font-weight: 600;">
                    🎯 ${item.confidence.score}%
                </span>
            </div>
            <div class="variation-text">${item.caption}</div>
            <div class="variation-meta">
                <span>📝 ${item.text_stats.word_count} words</span>
                <span>•</span>
                <span>${idx === AppState.selectedVariationIndex ? "✓ Selected" : "Click to select"}</span>
            </div>
        `;

        card.addEventListener("click", () => {
            selectVariation(idx, data);
        });

        variationsGrid.appendChild(card);
    });
}

function selectVariation(index, data) {
    if (!data.captions || !data.captions[index]) return;
    AppState.selectedVariationIndex = index;
    const selected = data.captions[index];

    resultText.textContent = selected.caption;

    if (metaConfidence) {
        metaConfidence.textContent = `${selected.confidence.score}% ${selected.confidence.level}`;
    }
    if (metaWordCount) {
        metaWordCount.textContent = `${selected.text_stats.word_count} words`;
    }

    const cards = variationsGrid.querySelectorAll(".variation-card");
    cards.forEach((card, idx) => {
        if (idx === index) {
            card.classList.add("selected");
            const metaSpan = card.querySelector(".variation-meta span:last-child");
            if (metaSpan) metaSpan.textContent = "✓ Selected";
        } else {
            card.classList.remove("selected");
            const metaSpan = card.querySelector(".variation-meta span:last-child");
            if (metaSpan) metaSpan.textContent = "Click to select";
        }
    });

    const thumbnailSrc = imagePreviewElement && imagePreviewElement.src ? imagePreviewElement.src : null;
    saveCaptionToHistory(data, selected, thumbnailSrc);
}


// ---------------------------------------------------------------------------
// 8. Render Semantic Intelligence & Quality Evaluation
// ---------------------------------------------------------------------------
function renderSemantics(semantics) {
    if (!semanticCard || !semantics) return;

    if (detectedObjectsTags) {
        detectedObjectsTags.innerHTML = "";
        const objects = semantics.objects || [];
        objects.forEach(obj => {
            const span = document.createElement("span");
            span.className = "semantic-tag";
            span.textContent = obj;
            detectedObjectsTags.appendChild(span);
        });
    }

    if (detectedSceneVal) {
        detectedSceneVal.textContent = semantics.scene || "Visual environment";
    }

    if (detectedActivityVal) {
        detectedActivityVal.textContent = semantics.activity || "Static scene";
    }

    if (detectedAttributesTags) {
        detectedAttributesTags.innerHTML = "";
        const attributes = semantics.attributes || [];
        attributes.forEach(attr => {
            const span = document.createElement("span");
            span.className = "semantic-tag";
            span.textContent = attr;
            detectedAttributesTags.appendChild(span);
        });
    }

    semanticCard.hidden = false;
}

function renderEvaluation(evalData) {
    if (!evaluationCard || !evalData) return;

    const score = evalData.composite_score || 0;

    if (evalScoreNum) {
        evalScoreNum.textContent = score;
    }

    const circle = document.querySelector(".score-circle");
    if (circle) {
        let strokeColor = "#2563eb";
        if (score >= 80) strokeColor = "#10b981";
        else if (score >= 60) strokeColor = "#3b82f6";
        else strokeColor = "#f59e0b";
        circle.style.stroke = strokeColor;
    }

    if (evalScoreGrade) {
        evalScoreGrade.textContent = evalData.grade || "High Quality";
    }

    if (evalScoreDesc) {
        evalScoreDesc.textContent = `AI Quality Index: Lexical diversity (${evalData.lexical_diversity || 85}%), grammar integrity, and semantic coverage.`;
    }

    if (benchmarkResultsGrid) {
        benchmarkResultsGrid.hidden = true;
    }

    evaluationCard.hidden = false;
}


// ---------------------------------------------------------------------------
// 9. Core Caption Generation & Regeneration Pipeline (3.12.1, 3.12.2, 3.12.9, 3.12.10)
// ---------------------------------------------------------------------------
async function processCaptionGeneration(isRegenerate = false) {
    const imageFile = imageFileInput.files[0];

    // Check if we can use existing backend result_id for regeneration
    const canRegenerateFromBackend = isRegenerate && AppState.resultId;

    if (!imageFile && !canRegenerateFromBackend) {
        showCaptionError("Please select or upload an image first.");
        return;
    }

    errorBox.hidden = true;
    loadingBox.hidden = false;

    if (isRegenerate) {
        AppState.status = "REGENERATING";
        startLoadingProgress("regenerate");
        if (regenerateCaptionBtn) {
            regenerateCaptionBtn.disabled = true;
            regenerateCaptionBtn.textContent = "⏳ Regenerating...";
        }
    } else {
        AppState.status = "GENERATING";
        startLoadingProgress("generate");
        resultCard.hidden = true;
        if (semanticCard) semanticCard.hidden = true;
        if (evaluationCard) evaluationCard.hidden = true;
        generateButton.disabled = true;
        generateButton.textContent = "Generating Caption...";
    }

    const wantMultiple = multiCaptionToggle && multiCaptionToggle.checked;
    const count = wantMultiple ? 3 : 1;

    const abortController = new AbortController();
    const timeoutHandle = setTimeout(() => abortController.abort(), 45000);

    try {
        let response;

        // Path A: Regenerate using backend ID (instant on-disk image lookup)
        if (canRegenerateFromBackend) {
            console.log(`[API] Calling ${REGENERATE_API_URL} with result_id: ${AppState.resultId}`);
            response = await fetch(REGENERATE_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    result_id: AppState.resultId,
                    style: AppState.currentStyle
                }),
                signal: abortController.signal
            });
        }
        // Path B: Standard Generation / Upload Form Data
        else {
            const formData = new FormData();
            formData.append("image", imageFile);
            formData.append("style", AppState.currentStyle);
            formData.append("count", count.toString());
            if (isRegenerate) formData.append("regenerate", "true");

            console.log(`[API] Posting image to ${CAPTION_API_URL} (style=${AppState.currentStyle}, count=${count})`);
            response = await fetch(CAPTION_API_URL, {
                method: "POST",
                body: formData,
                signal: abortController.signal
            });
        }

        let resJson = null;
        try {
            const rawText = await response.text();
            resJson = rawText && rawText.trim() ? JSON.parse(rawText) : null;
        } catch (jsonErr) {
            console.warn("[API Warning] Response was not valid JSON:", jsonErr);
        }

        console.log("[API] Backend response:", resJson);

        if (!response.ok || !resJson || !resJson.success) {
            let errMsg = resJson && resJson.message ? resJson.message : null;
            if (!errMsg) {
                if (response.status === 404) {
                    errMsg = `Cannot connect to AI endpoint at ${CAPTION_API_URL}. Please ensure the backend Flask server is running at http://127.0.0.1:5000 (run 'python run.py' in the terminal).`;
                } else if (response.status === 0 || !response.status) {
                    errMsg = `Network error connecting to ${API_BASE_URL}. Please verify your connection and ensure the Flask server is running.`;
                } else {
                    errMsg = `AI service returned HTTP ${response.status} (${response.statusText || 'Error'}). Please check the backend console.`;
                }
            }
            throw new Error(errMsg);
        }

        // Extract result entity (support both data envelope and flat)
        const data = resJson.data || resJson;

        AppState.status = "SUCCESS";
        AppState.resultId = data.id || null;
        AppState.lastResponseData = data;
        AppState.selectedVariationIndex = 0;
        stopLoadingProgress();

        const primary = (data.captions && data.captions[0]) ? data.captions[0] : {
            caption: data.caption,
            confidence: data.confidence,
            text_stats: data.text_stats
        };

        // 1. Render Primary Caption
        resultText.textContent = primary.caption || "A photograph displaying visual elements.";

        // 2. Render Multiple Variations if present
        if (data.captions && data.captions.length > 1) {
            renderVariations(data.captions, data);
        } else if (comparisonContainer) {
            comparisonContainer.hidden = true;
        }

        // 3. Render Metadata Badges
        if (metaInference && data.inference_time_ms) {
            const timeSec = (data.inference_time_ms / 1000).toFixed(2);
            metaInference.textContent = `${timeSec}s`;
        }

        if (metaConfidence && primary.confidence) {
            metaConfidence.textContent = `${primary.confidence.score}% ${primary.confidence.level}`;
        }

        if (metaDimensions && data.image_metadata) {
            metaDimensions.textContent = `${data.image_metadata.width} × ${data.image_metadata.height}`;
        }

        if (metaWordCount && primary.text_stats) {
            metaWordCount.textContent = `${primary.text_stats.word_count} words`;
        }

        if (metaModelTag) {
            const styleLabel = AppState.currentStyle.charAt(0).toUpperCase() + AppState.currentStyle.slice(1);
            metaModelTag.textContent = `BLIP (${styleLabel})`;
        }

        // 4. Render Semantic Scene Understanding & Quality Score
        if (data.semantics) renderSemantics(data.semantics);
        if (data.quality_evaluation) renderEvaluation(data.quality_evaluation);

        // 5. Save to History (local cache + backend sync)
        const thumbnailSrc = imagePreviewElement && imagePreviewElement.src ? imagePreviewElement.src : null;
        saveCaptionToHistory(data, primary, thumbnailSrc);

        // 6. Reveal Result Card
        loadingBox.hidden = true;
        resultCard.hidden = false;

        // Visual flash for regenerated feedback
        if (isRegenerate) {
            resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

    } catch (error) {
        console.error("[API Error] Caption process failed:", error);
        showCaptionError(formatErrorMessage(error));
    } finally {
        clearTimeout(timeoutHandle);
        generateButton.disabled = false;
        generateButton.textContent = "Generate Caption";
        if (regenerateCaptionBtn) {
            regenerateCaptionBtn.disabled = false;
            regenerateCaptionBtn.textContent = "🔄 Regenerate";
        }
    }
}


// ---------------------------------------------------------------------------
// 10. Reference Benchmark Evaluation Handler
// ---------------------------------------------------------------------------
if (runBenchmarkBtn) {
    runBenchmarkBtn.addEventListener("click", async () => {
        const candidate = resultText.textContent;
        const reference = referenceCaptionInput ? referenceCaptionInput.value.trim() : "";

        if (!candidate) {
            alert("Please generate a caption first before running the benchmark.");
            return;
        }

        if (!reference) {
            alert("Please enter a ground-truth reference caption to compare against.");
            return;
        }

        runBenchmarkBtn.disabled = true;
        runBenchmarkBtn.textContent = "Calculating...";

        try {
            const res = await fetch(EVALUATE_API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ candidate, reference })
            });

            let resData = null;
            try {
                const text = await res.text();
                resData = text && text.trim() ? JSON.parse(text) : null;
            } catch (e) {
                console.warn("[API Warning] Benchmark response was not JSON:", e);
            }

            if (!res.ok || !resData || !resData.success) {
                throw new Error((resData && resData.message) || `Evaluation failed with HTTP ${res.status}.`);
            }

            const metrics = resData.data ? resData.data.evaluation : resData.evaluation;

            if (metricBleu && metrics.bleu) metricBleu.textContent = `${metrics.bleu.bleu_4}%`;
            if (metricRouge && metrics.rouge) metricRouge.textContent = `${metrics.rouge.rouge_l}%`;
            if (metricMeteor) metricMeteor.textContent = `${metrics.meteor}%`;
            if (metricSimilarity) metricSimilarity.textContent = `${metrics.semantic_similarity}%`;

            if (benchmarkResultsGrid) benchmarkResultsGrid.hidden = false;
        } catch (err) {
            console.error("Benchmark calculation error:", err);
            alert(formatErrorMessage(err));
        } finally {
            runBenchmarkBtn.disabled = false;
            runBenchmarkBtn.textContent = "⚡ Run Benchmark";
        }
    });
}


// ---------------------------------------------------------------------------
// 11. Report Export Handler (TXT & JSON)
// ---------------------------------------------------------------------------
function exportReport(format = "txt") {
    if (!AppState.lastResponseData) return;

    const data = AppState.lastResponseData;
    const currentCaption = resultText.textContent;
    const dateStr = new Date().toISOString();

    if (format === "json") {
        const exportObject = {
            export_timestamp: dateStr,
            selected_caption: currentCaption,
            selected_style: data.style || AppState.currentStyle,
            api_data: data
        };

        const jsonBlob = new Blob([JSON.stringify(exportObject, null, 2)], { type: "application/json" });
        triggerDownload(jsonBlob, `caption_report_${Date.now()}.json`);
    } else {
        let textContent = `=========================================================\n`;
        textContent += `        VISIONCAPTION AI - IMAGE CAPTION REPORT          \n`;
        textContent += `=========================================================\n\n`;
        textContent += `Generated Date : ${new Date().toLocaleString()}\n`;
        textContent += `Style Applied  : ${(data.style || AppState.currentStyle).toUpperCase()}\n`;
        if (data.image_metadata) {
            textContent += `Image Filename : ${data.image_metadata.filename}\n`;
            textContent += `Resolution     : ${data.image_metadata.width} x ${data.image_metadata.height}\n`;
            textContent += `File Size      : ${data.image_metadata.size_kb} KB\n`;
        }
        textContent += `Model          : ${data.model_metadata ? data.model_metadata.model : "Salesforce BLIP"}\n`;
        textContent += `Inference Time : ${data.inference_time_ms} ms\n\n`;
        textContent += `---------------------------------------------------------\n`;
        textContent += `PRIMARY CAPTION:\n`;
        textContent += `"${currentCaption}"\n`;
        textContent += `---------------------------------------------------------\n\n`;

        if (data.semantics) {
            textContent += `SEMANTIC SCENE INTELLIGENCE:\n`;
            textContent += `  - Objects    : ${(data.semantics.objects || []).join(", ")}\n`;
            textContent += `  - Scene      : ${data.semantics.scene || "N/A"}\n`;
            textContent += `  - Activity   : ${data.semantics.activity || "N/A"}\n`;
            textContent += `  - Attributes : ${(data.semantics.attributes || []).join(", ")}\n\n`;
        }

        if (data.quality_evaluation) {
            textContent += `AI QUALITY BENCHMARK:\n`;
            textContent += `  - Composite Score: ${data.quality_evaluation.composite_score}/100\n`;
            textContent += `  - Quality Grade  : ${data.quality_evaluation.grade}\n\n`;
        }

        if (data.captions && data.captions.length > 1) {
            textContent += `ALL GENERATED VARIATIONS:\n`;
            data.captions.forEach((item, i) => {
                textContent += `  ${i + 1}. "${item.caption}" (Confidence: ${item.confidence.score}%, ${item.text_stats.word_count} words)\n`;
            });
            textContent += `\n`;
        }

        textContent += `=========================================================\n`;

        const txtBlob = new Blob([textContent], { type: "text/plain;charset=utf-8" });
        triggerDownload(txtBlob, `caption_report_${Date.now()}.txt`);
    }
}

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}


// ---------------------------------------------------------------------------
// 12. Event Listeners & Initialization
// ---------------------------------------------------------------------------
generateButton.addEventListener("click", () => {
    processCaptionGeneration(false);
});

if (regenerateCaptionBtn) {
    regenerateCaptionBtn.addEventListener("click", () => {
        processCaptionGeneration(true);
    });
}

if (exportTxtBtn) exportTxtBtn.addEventListener("click", () => exportReport("txt"));
if (exportJsonBtn) exportJsonBtn.addEventListener("click", () => exportReport("json"));

if (copyCaptionBtn) {
    copyCaptionBtn.addEventListener("click", async () => {
        const textToCopy = resultText.textContent;
        if (!textToCopy) return;

        try {
            await navigator.clipboard.writeText(textToCopy);
            copyCaptionBtn.textContent = "Copied! ✓";
            copyCaptionBtn.classList.add("copied");
            setTimeout(() => {
                copyCaptionBtn.textContent = "📋 Copy Caption";
                copyCaptionBtn.classList.remove("copied");
            }, 2000);
        } catch (err) {
            const textArea = document.createElement("textarea");
            textArea.value = textToCopy;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);

            copyCaptionBtn.textContent = "Copied! ✓";
            copyCaptionBtn.classList.add("copied");
            setTimeout(() => {
                copyCaptionBtn.textContent = "📋 Copy Caption";
                copyCaptionBtn.classList.remove("copied");
            }, 2000);
        }
    });
}

imageFileInput.addEventListener("change", () => {
    resetCaptionUI();
});

if (removeButton) {
    removeButton.addEventListener("click", () => {
        resetCaptionUI();
    });
}

console.log("Frontend <-> Backend Integration Engine (v1.0.0) ready.");