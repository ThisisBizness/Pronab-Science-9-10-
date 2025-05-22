document.addEventListener('DOMContentLoaded', () => {
    const questionForm = document.getElementById('question-form');
    const userTextQuestionInput = document.getElementById('user-text-question');
    const userImageQuestionInput = document.getElementById('user-image-question');
    const submitButton = document.getElementById('submit-button');
    const answerBox = document.getElementById('answer-box');
    const answerSection = document.getElementById('answer-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessageDiv = document.getElementById('error-message');
    const imagePreview = document.getElementById('image-preview');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const removeImageButton = document.getElementById('remove-image-button');

    let currentSessionId = sessionStorage.getItem('scienceAppSessionId');
    if (!currentSessionId) {
        // Simple way to generate a pseudo-UUID for client-side session tracking if needed
        currentSessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        sessionStorage.setItem('scienceAppSessionId', currentSessionId);
    }

    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'flex' : 'none';
        submitButton.disabled = isLoading;
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
        formattedText = formattedText.replace(/^\s*[\*\-]\s+(.*)/gm, '<li>$1</li>');
        const listRegex = /(<li>.*<\/li>)/s;
        if (listRegex.test(formattedText)) {
            formattedText = formattedText.replace(listRegex, '<ul>$1</ul>');
        }
        formattedText = formattedText.replace(/<\/ul>\s*<ul>/g, ''); // Clean up multiple wraps
        return formattedText;
    }

    userImageQuestionInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreviewContainer.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = "#";
            imagePreviewContainer.style.display = 'none';
        }
    });

    removeImageButton.addEventListener('click', function() {
        userImageQuestionInput.value = ''; // Clear the file input
        imagePreview.src = "#";
        imagePreviewContainer.style.display = 'none';
    });


    questionForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Prevent default form submission

        const textQuestion = userTextQuestionInput.value.trim();
        const imageFile = userImageQuestionInput.files[0];

        if (!textQuestion && !imageFile) {
            displayError('Please type a question or upload an image.');
            return;
        }

        showLoading(true);

        const formData = new FormData();
        formData.append('session_id', currentSessionId);
        if (textQuestion) {
            formData.append('text_question', textQuestion);
        }
        if (imageFile) {
            formData.append('image_question', imageFile);
        }

        try {
            const response = await fetch('/ask-science', {
                method: 'POST',
                body: formData, // FormData will set Content-Type to multipart/form-data
            });

            showLoading(false);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
                displayError(`Error ${response.status}: ${errorData.detail}`);
                return;
            }

            const data = await response.json();

            if (data.session_id && data.session_id !== currentSessionId) {
                currentSessionId = data.session_id;
                sessionStorage.setItem('scienceAppSessionId', currentSessionId);
            }

            answerBox.innerHTML = formatAnswer(data.answer);
            answerSection.style.display = 'block';
            errorMessageDiv.style.display = 'none'; // Clear previous errors

            // Clear inputs after successful submission
            // userTextQuestionInput.value = '';
            // userImageQuestionInput.value = ''; // Clears the file input
            // imagePreview.src = "#";
            // imagePreviewContainer.style.display = 'none';

        } catch (error) {
            showLoading(false);
            console.error('Fetch error:', error);
            displayError('There was a problem connecting to the server. Please try again.');
        }
    });
}); 