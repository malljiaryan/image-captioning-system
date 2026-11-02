/* =========================================
   VISIONCAPTION AI - SHARED APPLICATION JS
========================================= */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Mobile Hamburger Menu Toggle
    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.getElementById("navLinks");
    const mainNavbar = document.getElementById("mainNavbar");

    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", (e) => {
            e.stopPropagation();
            menuToggle.classList.toggle("active");
            navLinks.classList.toggle("open");
        });

        // Close mobile menu when clicking outside
        document.addEventListener("click", (e) => {
            if (!navLinks.contains(e.target) && !menuToggle.contains(e.target)) {
                menuToggle.classList.remove("active");
                navLinks.classList.remove("open");
            }
        });

        // Close mobile menu when clicking any nav link
        navLinks.querySelectorAll("a").forEach(link => {
            link.addEventListener("click", () => {
                menuToggle.classList.remove("active");
                navLinks.classList.remove("open");
            });
        });
    }

    // 2. Navbar Scroll Elevation
    if (mainNavbar) {
        window.addEventListener("scroll", () => {
            if (window.scrollY > 15) {
                mainNavbar.classList.add("scrolled");
            } else {
                mainNavbar.classList.remove("scrolled");
            }
        }, { passive: true });
    }
});

/**
 * Universal Toast Notification Helper
 * @param {string} message - Message text
 * @param {'success'|'error'|'info'} type - Type of toast
 * @param {number} durationMs - How long before dismissal
 */
function showToast(message, type = "success", durationMs = 3200) {
    let container = document.querySelector(".toast-container");
    if (!container) {
        container = document.createElement("div");
        container.className = "toast-container";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icon = type === "success" ? "✓" : (type === "error" ? "⚠️" : "ℹ️");
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
    `;

    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    // Animate out & remove
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 350);
    }, durationMs);
}
