document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const videoInput = document.getElementById('videoUrl');
    const statusText = document.getElementById('statusText');
    const resultContent = document.getElementById('resultContent');
    const welcomeState = document.getElementById('welcomeState');
    const errorState = document.getElementById('errorState');
    const errorMessage = document.getElementById('errorMessage');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');

    generateBtn.addEventListener('click', async () => {
        const url = videoInput.value.trim();
        
        if (!url) {
            alert("Please enter a YouTube URL");
            return;
        }

        // 1. Reset UI State
        setLoading(true);
        errorState.classList.add('hidden');
        welcomeState.classList.add('hidden');
        resultContent.classList.add('hidden');
        
        try {
            // 2. Start the Agent Pipeline
            updateStatus("Agent 1/3: Analyzing Video Transcript...");
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ video_url: url })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to generate blog");
            }

            // 3. Render Output
            updateStatus("Agents Finished!");
            // Use marked.js to parse the markdown string into HTML
            resultContent.innerHTML = marked.parse(data.blog_post);
            resultContent.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            errorMessage.textContent = error.message;
            errorState.classList.remove('hidden');
            updateStatus("Error Occurred");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        generateBtn.disabled = isLoading;
        if (isLoading) {
            btnText.classList.add('hidden');
            btnLoader.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
        }
    }

    function updateStatus(text) {
        statusText.textContent = text;
        
        // Simulating progress updates for better UX since backend is blocking
        if (text.includes("Analyzing")) {
            setTimeout(() => {
                if(generateBtn.disabled) statusText.textContent = "Agent 2/3: Researching Web Context...";
            }, 8000); // Guessing time for step 1
            setTimeout(() => {
                if(generateBtn.disabled) statusText.textContent = "Agent 3/3: Drafting Blog Post...";
            }, 18000); // Guessing time for step 2
        }
    }
});