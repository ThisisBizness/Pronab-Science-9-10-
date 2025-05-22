document.addEventListener('DOMContentLoaded', () => {
    const userQuestionInput = document.getElementById('user-question');
    const imageUploadInput = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const removeImageButton = document.getElementById('remove-image-button');
    
    const submitButton = document.getElementById('submit-button');
    const regenerateButton = document.getElementById('regenerate-button');
    const simplifyButton = document.getElementById('simplify-button');
    
    const answerBox = document.getElementById('answer-box');
    const answerSection = document.getElementById('answer-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');

    let currentSessionId = sessionStorage.getItem('scienceHelperSessionId');
    // We don't store lastQuestion client-side as much; backend context is more robust.

    // Image Preview Logic
    imageUploadInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '#';
            imagePreviewContainer.style.display = 'none';
        }
    });

    removeImageButton.addEventListener('click', () => {
        imageUploadInput.value = ''; // Clear the file input
        imagePreview.src = '#';
        imagePreviewContainer.style.display = 'none';
    });


    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'flex' : 'none';
        submitButton.disabled = isLoading;
        regenerateButton.disabled = isLoading || !answerBox.innerHTML.trim();
        simplifyButton.disabled = isLoading || !answerBox.innerHTML.trim();
        if (isLoading) {
            answerSection.style.display = 'none';
            errorMessageDiv.style.display = 'none';
        }
    }

    function displayError(message) {
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
        answerSection.style.display = 'none';
    }
    
    function formatAnswer(text) {
        // Basic formatting: replace newlines with <br>
        let formattedText = text.replace(/\n/g, '<br>');
        // Make bold text actually bold (handles **text**)
        formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        // Handle bullet points (simple * or - list items)
        formattedText = formattedText.replace(/^[\t ]*[\*\-][\t ]+(.*)/gm, '<li>$1</li>');
        const listRegex = /(<li>.*<\/li>)/s; 
        if (listRegex.test(formattedText)) {
            let tempFormattedText = formattedText;
            // Wrap consecutive list items in <ul> tags
            tempFormattedText = tempFormattedText.replace(/(<li>.*?<\/li>(?:<br>)*)+/g, (match) => `<ul>${match.replace(/<br>/g,'')}</ul>`);
             // Clean up potential multiple wraps or empty <ul> tags
            tempFormattedText = tempFormattedText.replace(/<\/ul>\s*<ul>/g, '');
            formattedText = tempFormattedText;
        }
        return formattedText;
    }


    async function handleSubmit(actionType) {
        const questionText = userQuestionInput.value.trim();
        const imageFile = imageUploadInput.files[0];

        if (actionType === 'ask' && !questionText && !imageFile) {
            displayError('Please type a question or upload an image.');
            return;
        }

        showLoading(true);

        const formData = new FormData();
        if (currentSessionId) {
            formData.append('session_id', currentSessionId);
        }
        if (questionText) {
            formData.append('question', questionText);
        }
        if (imageFile && actionType === 'ask') { // Only send image on initial "ask"
            formData.append('image', imageFile);
        }
        formData.append('action', actionType);
        
        // For regenerate/simplify, the backend uses stored context.
        // No need to resend image unless explicitly designed for it.

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                body: formData, // No 'Content-Type' header for FormData; browser sets it.
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                displayError(`Error ${response.status}: ${errorData.detail || 'Failed to get response.'}`);
                showLoading(false);
                return;
            }

            const data = await response.json();
            showLoading(false);

            if (data.session_id && data.session_id !== currentSessionId) {
                currentSessionId = data.session_id;
                sessionStorage.setItem('scienceHelperSessionId', currentSessionId);
            }

            answerBox.innerHTML = formatAnswer(data.answer);
            answerSection.style.display = 'block';
            errorMessageDiv.style.display = 'none';

            // Enable action buttons
            regenerateButton.disabled = false;
            simplifyButton.disabled = false;

            if (actionType === 'ask') {
                // Optionally clear inputs after successful "ask"
                // userQuestionInput.value = '';
                // imageUploadInput.value = '';
                // imagePreview.src = '#';
                // imagePreviewContainer.style.display = 'none';
            }

        } catch (error) {
            showLoading(false);
            console.error('Fetch error:', error);
            displayError('Could not connect to the server. Please check your connection.');
        }
    }

    submitButton.addEventListener('click', () => handleSubmit('ask'));
    regenerateButton.addEventListener('click', () => handleSubmit('regenerate'));
    simplifyButton.addEventListener('click', () => handleSubmit('simplify'));
    
    // Initially disable action buttons
    regenerateButton.disabled = true;
    simplifyButton.disabled = true;
}); 