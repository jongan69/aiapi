// DOM Elements
const toggleButton = document.querySelector('.toggle-button');
const chatContainer = document.querySelector('.chat-container');
const swaggerContainer = document.querySelector('.swagger-container');
const chatModelSelect = document.querySelector('#model-select');
const imageModelSelect = document.querySelector('#image-model');
const chatMessages = document.querySelector('.chat-messages');
const chatInput = document.querySelector('.chat-textarea');
const sendButton = document.querySelector('.send-button');
const imagePromptInput = document.querySelector('.image-prompt-input');
const generateButton = document.querySelector('.generate-button');
const generatedImage = document.querySelector('.generated-image');
const typingIndicator = document.querySelector('.typing-indicator');
const jsonModeToggle = document.querySelector('#json-mode-toggle');
const variationButton = document.querySelector('.variation-button');
const variationModal = document.querySelector('#variation-modal');
const closeModal = document.querySelector('.close-modal');
const imageUpload = document.querySelector('#image-upload');
const imagePreview = document.querySelector('#image-preview');
const generateVariationsButton = document.querySelector('.generate-variations-button');
const variationResults = document.querySelector('.variation-results');

// State
let isChatMode = false;
let isProcessing = false;
let availableModels = [];
let availableImageModels = [];

// Initialize
async function init() {
    try {
        await Promise.all([fetchModels(), fetchImageModels()]);
        setupEventListeners();
        setupAutoResize();
    } catch (error) {
        console.error('Error during initialization:', error);
    }
}

// Fetch available models
async function fetchModels() {
    try {
        const response = await fetch('/models/');
        const data = await response.json();
        availableModels = data.models;
        populateChatModelSelect();
    } catch (error) {
        console.error('Error fetching models:', error);
        showError('Failed to fetch available models');
    }
}

// Fetch available image models
async function fetchImageModels() {
    try {
        const response = await fetch('/models/image/');
        const data = await response.json();
        availableImageModels = data.models;
        populateImageModelSelect();
    } catch (error) {
        console.error('Error fetching image models:', error);
        showError('Failed to fetch available image models');
    }
}

// Populate chat model select dropdown
function populateChatModelSelect() {
    if (!chatModelSelect) {
        console.error('Chat model select element not found');
        return;
    }
    
    chatModelSelect.innerHTML = availableModels
        .map(model => `<option value="${model}">${model}</option>`)
        .join('');
}

// Populate image model select dropdown
function populateImageModelSelect() {
    if (!imageModelSelect) {
        console.error('Image model select element not found');
        return;
    }
    
    imageModelSelect.innerHTML = availableImageModels
        .map(model => `<option value="${model}">${model}</option>`)
        .join('');
}

// Setup event listeners
function setupEventListeners() {
    toggleButton.addEventListener('click', toggleUI);
    sendButton.addEventListener('click', handleSend);
    chatInput.addEventListener('keypress', handleKeyPress);
    generateButton.addEventListener('click', handleImageGeneration);
    variationButton.addEventListener('click', openVariationModal);
    closeModal.addEventListener('click', closeVariationModal);
    imageUpload.addEventListener('change', handleImageUpload);
    generateVariationsButton.addEventListener('click', handleImageVariations);
    
    // Update button text on JSON mode toggle
    jsonModeToggle.addEventListener('change', () => {
        chatInput.placeholder = jsonModeToggle.checked ? 
            'Enter your message (responses will be in JSON format)...' : 
            'Type your message...';
    });
}

// Setup textarea auto-resize
function setupAutoResize() {
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
}

// Toggle between chat and Swagger UI
function toggleUI() {
    isChatMode = !isChatMode;
    chatContainer.style.display = isChatMode ? 'block' : 'none';
    swaggerContainer.style.display = isChatMode ? 'none' : 'block';
    toggleButton.textContent = isChatMode ? 'Show API Docs' : 'Show Chat';
    
    if (isChatMode) {
        chatInput.focus();
    }
}

// Show/hide typing indicator
function toggleTypingIndicator(show) {
    typingIndicator.style.display = show ? 'flex' : 'none';
    if (show) {
        typingIndicator.classList.add('visible');
    } else {
        typingIndicator.classList.remove('visible');
    }
}

function formatJSON(json) {
    const formatted = JSON.stringify(json, null, 2);
    return formatted.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function appendMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    
    if (isUser) {
        messageDiv.textContent = content;
    } else {
        try {
            // Check if content is a JSON string
            if (typeof content === 'string' && (content.trim().startsWith('{') || content.trim().startsWith('['))) {
                try {
                    const jsonContent = JSON.parse(content);
                    const pre = document.createElement('pre');
                    pre.className = 'json-content';
                    pre.innerHTML = formatJSON(jsonContent);
                    messageDiv.appendChild(pre);
                } catch (e) {
                    // If not valid JSON, check for markdown code blocks
                    if (content.includes('```')) {
                        const parts = content.split('```');
                        parts.forEach((part, index) => {
                            if (index % 2 === 0) {
                                // Regular text
                                const textDiv = document.createElement('div');
                                textDiv.className = 'markdown-content';
                                textDiv.textContent = part;
                                messageDiv.appendChild(textDiv);
                            } else {
                                // Code block
                                const [lang, ...code] = part.split('\n');
                                const codeBlock = document.createElement('pre');
                                codeBlock.className = 'code-block';
                                if (lang) {
                                    const langLabel = document.createElement('div');
                                    langLabel.className = 'code-language';
                                    langLabel.textContent = lang.trim();
                                    codeBlock.appendChild(langLabel);
                                }
                                const codeContent = document.createElement('code');
                                if (lang === 'json') {
                                    try {
                                        const jsonContent = JSON.parse(code.join('\n'));
                                        codeContent.innerHTML = formatJSON(jsonContent);
                                    } catch (e) {
                                        codeContent.textContent = code.join('\n');
                                    }
                                } else {
                                    codeContent.textContent = code.join('\n');
                                }
                                codeBlock.appendChild(codeContent);
                                messageDiv.appendChild(codeBlock);
                            }
                        });
                    } else {
                        messageDiv.textContent = content;
                    }
                }
            } else {
                messageDiv.textContent = content;
            }
        } catch (e) {
            messageDiv.textContent = content;
        }
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error-message';
    errorDiv.textContent = message;
    chatMessages.appendChild(errorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Handle send button click
async function handleSend() {
    if (isProcessing) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    isProcessing = true;
    sendButton.disabled = true;
    toggleTypingIndicator(true);
    
    appendMessage(message, true);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    try {
        const response = await fetch('/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{
                    role: "user",
                    content: message
                }],
                model: chatModelSelect.value,
                stream: true,
                json_mode: jsonModeToggle.checked
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Create a message div for the assistant's response
        const assistantMessageDiv = document.createElement('div');
        assistantMessageDiv.className = 'message assistant-message';
        chatMessages.appendChild(assistantMessageDiv);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let accumulatedContent = '';
        let hasShownError = false;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        break;
                    }

                    try {
                        const parsed = JSON.parse(data);
                        
                        if (parsed.error && !hasShownError) {
                            showError(parsed.error);
                            hasShownError = true;
                            break;
                        }

                        if (jsonModeToggle.checked) {
                            // For JSON mode, try to parse the content as JSON
                            if (parsed.content) {
                                try {
                                    const jsonContent = JSON.parse(parsed.content);
                                    // If it's a complete JSON object, display it formatted
                                    assistantMessageDiv.innerHTML = formatJSON(jsonContent);
                                } catch (e) {
                                    // If not valid JSON yet, accumulate the content
                                    accumulatedContent += parsed.content;
                                    assistantMessageDiv.textContent = accumulatedContent;
                                }
                            }
                        } else {
                            // For non-JSON mode, just accumulate the content
                            if (parsed.content) {
                                accumulatedContent += parsed.content;
                                assistantMessageDiv.textContent = accumulatedContent;
                            }
                        }
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    } catch (e) {
                        // Only show parsing errors if we haven't shown one yet
                        if (!hasShownError) {
                            console.error('Error parsing SSE data:', e);
                            showError('Error parsing response');
                            hasShownError = true;
                        }
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error sending message:', error);
        showError('Failed to send message. Please try again.');
    } finally {
        isProcessing = false;
        sendButton.disabled = false;
        toggleTypingIndicator(false);
    }
}

// Handle key press
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSend();
    }
}

// Handle image generation
async function handleImageGeneration() {
    const prompt = imagePromptInput.value.trim();
    if (!prompt) return;
    
    generateButton.disabled = true;
    const originalText = generateButton.innerHTML;
    generateButton.innerHTML = '<span class="button-text">Generating...</span><span class="button-icon">✨</span>';
    
    try {
        const response = await fetch('/images/generate/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                model: imageModelSelect.value
            }),
        });

        const data = await response.json();
        
        if (data.url) {
            const img = document.createElement('img');
            img.src = data.url;
            img.alt = prompt;
            generatedImage.innerHTML = '';
            generatedImage.appendChild(img);
        } else {
            showError('Failed to generate image');
        }
    } catch (error) {
        console.error('Error generating image:', error);
        showError('Failed to generate image. Please try again.');
    } finally {
        generateButton.disabled = false;
        generateButton.innerHTML = originalText;
    }
}

// Open variation modal
function openVariationModal() {
    variationModal.style.display = 'block';
}

// Close variation modal
function closeVariationModal() {
    variationModal.style.display = 'none';
    imagePreview.style.display = 'none';
    imageUpload.value = '';
    variationResults.innerHTML = '';
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

// Handle image variations
async function handleImageVariations() {
    const fileInput = document.getElementById('image-upload');
    const modelSelect = document.getElementById('variation-model');
    const generateButton = document.querySelector('.generate-variations-button');
    const resultsContainer = document.querySelector('.variation-results');

    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an image first');
        return;
    }

    try {
        // Disable button and show loading state
        generateButton.disabled = true;
        generateButton.textContent = 'Generating...';
        resultsContainer.innerHTML = '<p>Generating variations...</p>';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model', modelSelect.value);
        formData.append('response_format', 'url');

        const response = await fetch('/images/variations/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to generate variations');
        }

        const data = await response.json();
        
        // Create the comparison container
        resultsContainer.innerHTML = `
            <div class="image-comparison-container">
                <div class="image-preview-container">
                    <div class="image-label">Original Image</div>
                    <div class="image-preview">
                        <img src="${URL.createObjectURL(fileInput.files[0])}" alt="Original image">
                    </div>
                </div>
                <div class="variation-preview-container">
                    <div class="image-label">Generated Variation</div>
                    <div class="variation-image">
                        ${data.url ? 
                            `<img src="${data.url}" alt="Generated variation">` :
                            data.b64_json ? 
                            `<img src="data:image/png;base64,${data.b64_json}" alt="Generated variation">` :
                            ''
                        }
                    </div>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error generating variations:', error);
        resultsContainer.innerHTML = `
            <div class="error-message">
                <p>Error: ${error.message}</p>
                <p>Please try again with a different image or model.</p>
            </div>
        `;
    } finally {
        // Re-enable button and restore text
        generateButton.disabled = false;
        generateButton.textContent = 'Generate Variations';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', init);
