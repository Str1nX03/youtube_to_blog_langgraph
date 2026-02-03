document.addEventListener('DOMContentLoaded', () => {
    console.log("Landing page loaded");

    const btn = document.getElementById('getStartedBtn');
    
    // Simple interactive hover effect
    btn.addEventListener('mouseenter', () => {
        btn.style.boxShadow = "0 0 20px rgba(59, 130, 246, 0.6)";
    });

    btn.addEventListener('mouseleave', () => {
        btn.style.boxShadow = "none";
    });
});