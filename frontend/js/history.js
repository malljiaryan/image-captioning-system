/* =========================================
   HISTORY PAGE - BACKEND API & LOCAL STORAGE SYNC
   Milestone 3.11 & 3.12 Full Integration + Modern Search & Filters
========================================= */

function getApiBaseUrl() {
    if (window.location.origin && window.location.origin.startsWith("http")) {
        if (window.location.port === "5000" || window.location.port === "" || window.location.port === "80" || window.location.port === "443") {
            return window.location.origin;
        }
        return `${window.location.protocol}//${window.location.hostname}:5000`;
    }
    return "http://127.0.0.1:5000";
}

const API_BASE = getApiBaseUrl();
const HISTORY_STORAGE_KEY = "visionCaptionHistory";
const HISTORY_API_URL = `${API_BASE}/api/history`;

/* =========================================
   DOM ELEMENTS
========================================= */
const historyList = document.getElementById("historyList");
const historyEmpty = document.getElementById("historyEmpty");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const historyCount = document.querySelector(".history-count strong");
const searchInput = document.getElementById("historySearchInput");
const filterChips = document.querySelectorAll(".filter-chip");

let allHistoryItems = [];
let currentFilterStyle = "all";
let currentSearchQuery = "";

/* =========================================
   FETCH HISTORY (API + LOCAL SYNC)
========================================= */
async function fetchBackendHistory() {
    try {
        const response = await fetch(HISTORY_API_URL);
        if (!response.ok) return null;
        let result = null;
        try {
            const text = await response.text();
            result = text && text.trim() ? JSON.parse(text) : null;
        } catch (e) {
            console.warn("[History Warning] Response was not valid JSON:", e);
            return null;
        }
        if (result.success && result.data && Array.isArray(result.data.items)) {
            return result.data.items.map(item => ({
                id: item.id,
                icon: "✨",
                caption: item.caption,
                style: item.style || "standard",
                date: item.display_date || (item.created_at ? new Date(item.created_at).toLocaleString() : "Recently"),
                confidence: item.confidence ? `${item.confidence.score}% (${item.confidence.level})` : null,
                dimensions: item.image_metadata ? `${item.image_metadata.width}×${item.image_metadata.height}` : null,
                inferenceTime: item.inference_time_ms ? `${(item.inference_time_ms / 1000).toFixed(2)}s` : null,
                thumbnail: null,
                source: "backend"
            }));
        }
    } catch (err) {
        console.warn("Backend /api/history unreachable, using localStorage fallback:", err);
    }
    return null;
}

function getLocalHistory() {
    const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!saved) return [];
    try {
        return JSON.parse(saved);
    } catch (e) {
        return [];
    }
}

async function loadUnifiedHistory() {
    const localItems = getLocalHistory();
    const backendItems = await fetchBackendHistory();

    if (backendItems && backendItems.length > 0) {
        const localThumbMap = new Map();
        localItems.forEach(it => {
            if (it.caption && it.thumbnail) {
                localThumbMap.set(it.caption, it.thumbnail);
            }
        });

        return backendItems.map(bItem => {
            if (localThumbMap.has(bItem.caption)) {
                bItem.thumbnail = localThumbMap.get(bItem.caption);
            }
            return bItem;
        });
    }

    return localItems;
}

/* =========================================
   RENDER FILTERED HISTORY
========================================= */
function renderFilteredHistory() {
    let filtered = allHistoryItems.slice();

    // 1. Filter by Style Chip
    if (currentFilterStyle !== "all") {
        filtered = filtered.filter(item => (item.style || "standard").toLowerCase() === currentFilterStyle.toLowerCase());
    }

    // 2. Filter by Search Query
    if (currentSearchQuery.trim()) {
        const query = currentSearchQuery.toLowerCase();
        filtered = filtered.filter(item => 
            (item.caption && item.caption.toLowerCase().includes(query)) ||
            (item.style && item.style.toLowerCase().includes(query)) ||
            (item.date && item.date.toLowerCase().includes(query))
        );
    }

    historyList.innerHTML = "";

    if (historyCount) {
        historyCount.textContent = filtered.length;
    }

    if (filtered.length === 0) {
        historyList.hidden = true;
        historyEmpty.hidden = false;
        return;
    }

    historyList.hidden = false;
    historyEmpty.hidden = true;

    filtered.forEach(item => {
        const historyCard = document.createElement("div");
        historyCard.className = "history-card";
        historyCard.dataset.id = item.id;

        const imageContent = item.thumbnail
            ? `<img src="${item.thumbnail}" alt="Thumbnail" style="width:100%;height:100%;object-fit:cover;border-radius:10px;">`
            : `<div class="history-image-placeholder">${item.icon || "🖼️"}</div>`;

        const metaBadges = [];
        if (item.style) metaBadges.push(`✦ ${item.style.toUpperCase()}`);
        if (item.confidence) metaBadges.push(`🎯 ${item.confidence}`);
        if (item.dimensions) metaBadges.push(`📐 ${item.dimensions}`);
        if (item.inferenceTime) metaBadges.push(`⚡ ${item.inferenceTime}`);

        const metaHtml = metaBadges.length > 0
            ? `<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;">${metaBadges.map(b => `<span style="font-size:11px;background:#f1f5f9;color:#475569;padding:3px 10px;border-radius:12px;font-weight:600;border:1px solid #e2e8f0;">${b}</span>`).join("")}</div>`
            : "";

        historyCard.innerHTML = `
            <div class="history-image">
                ${imageContent}
            </div>

            <div class="history-content">
                <span class="history-badge">AI GENERATED</span>
                <p class="history-caption">${item.caption}</p>
                <div style="display:flex;align-items:center;gap:12px;margin-top:4px;">
                    <span class="history-date">${item.date}</span>
                    <button type="button" class="quick-copy-btn" data-caption="${escapeHtml(item.caption)}" style="font-size:12px;color:var(--primary);font-weight:600;cursor:pointer;background:none;border:none;padding:0;">📋 Copy</button>
                </div>
                ${metaHtml}
            </div>

            <button
                type="button"
                class="history-delete-btn"
                aria-label="Delete caption"
                title="Delete from history"
            >
                ×
            </button>
        `;

        historyList.appendChild(historyCard);
    });
}

function escapeHtml(text) {
    if (!text) return "";
    return text.replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

async function reloadHistory() {
    allHistoryItems = await loadUnifiedHistory();
    if (clearHistoryBtn) {
        clearHistoryBtn.hidden = allHistoryItems.length === 0;
    }
    renderFilteredHistory();
}

/* =========================================
   SEARCH & FILTER LISTENERS
========================================= */
if (searchInput) {
    searchInput.addEventListener("input", (e) => {
        currentSearchQuery = e.target.value;
        renderFilteredHistory();
    });
}

if (filterChips) {
    filterChips.forEach(chip => {
        chip.addEventListener("click", () => {
            filterChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            currentFilterStyle = chip.dataset.filter || "all";
            renderFilteredHistory();
        });
    });
}

/* =========================================
   DELEGATED ACTIONS (DELETE & QUICK COPY)
========================================= */
historyList.addEventListener("click", async function (event) {
    // 1. Quick Copy
    const copyBtn = event.target.closest(".quick-copy-btn");
    if (copyBtn) {
        const caption = copyBtn.dataset.caption;
        if (caption) {
            try {
                await navigator.clipboard.writeText(caption);
                if (typeof showToast === "function") showToast("Caption copied to clipboard!", "success");
                copyBtn.textContent = "Copied! ✓";
                setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 1800);
            } catch (err) {
                console.error("Clipboard copy failed:", err);
            }
        }
        return;
    }

    // 2. Delete Single Item
    const deleteButton = event.target.closest(".history-delete-btn");
    if (!deleteButton) return;

    const historyCard = deleteButton.closest(".history-card");
    if (!historyCard) return;

    const itemId = historyCard.dataset.id;

    if (itemId && itemId.startsWith("cap_")) {
        try {
            await fetch(`${HISTORY_API_URL}/${itemId}`, { method: "DELETE" });
        } catch (e) {
            console.warn("Could not delete from backend API:", e);
        }
    }

    let local = getLocalHistory();
    local = local.filter(item => String(item.id) !== String(itemId));
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(local));

    if (typeof showToast === "function") showToast("Caption record deleted.", "info");
    await reloadHistory();
});

/* =========================================
   CLEAR ALL HISTORY
========================================= */
if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", async function () {
        const confirmClear = confirm("Are you sure you want to permanently clear all caption history?");
        if (!confirmClear) return;

        try {
            await fetch(HISTORY_API_URL, { method: "DELETE" });
        } catch (e) {
            console.warn("Could not clear history on backend:", e);
        }

        localStorage.removeItem(HISTORY_STORAGE_KEY);
        if (typeof showToast === "function") showToast("All caption history cleared.", "info");
        await reloadHistory();
    });
}

/* =========================================
   INITIALIZE
========================================= */
reloadHistory();

