// Authentication State
let authToken = null;
let refreshToken = null;
let currentUser = null;
let isRefreshing = false;

// Chat History State
let chatHistory = [];

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    // Check for tokens in URL (from OAuth callback)
    const urlParams = new URLSearchParams(window.location.search);
    const accessToken = urlParams.get('access_token');
    const refreshTokenParam = urlParams.get('refresh_token');
    const error = urlParams.get('error');
    
    if (accessToken && refreshTokenParam) {
        // Store both tokens
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('refresh_token', refreshTokenParam);
        authToken = accessToken;
        refreshToken = refreshTokenParam;
        // Clean URL
        window.history.replaceState({}, document.title, '/');
        await loadUser();
    } else if (error) {
        alert(`Authentication failed: ${urlParams.get('message') || 'Unknown error'}`);
        window.history.replaceState({}, document.title, '/');
        showLoginModal();
    } else {
        // Check for existing tokens
        authToken = localStorage.getItem('access_token');
        refreshToken = localStorage.getItem('refresh_token');
        if (authToken && refreshToken) {
            await loadUser();
        } else {
            showLoginModal();
        }
    }
    
    // Load chat history from localStorage
    const savedHistory = localStorage.getItem('chat_history');
    if (savedHistory) {
        try {
            chatHistory = JSON.parse(savedHistory);
            displayChatHistory();
        } catch (e) {
            console.error('Failed to load chat history:', e);
        }
    }
});

async function loadUser() {
    try {
        const response = await fetch('/api/v1/auth/me', {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });
        
        if (response.ok) {
            currentUser = await response.json();
            showApp();
        } else if (response.status === 401) {
            // Access token expired - try to refresh
            const refreshed = await tryRefreshToken();
            if (refreshed) {
                // Retry loading user with new token
                await loadUser();
            } else {
                // Refresh failed - redirect to login
                clearTokens();
                showLoginModal();
            }
        } else {
            clearTokens();
            showLoginModal();
        }
    } catch (error) {
        console.error('Error loading user:', error);
        showLoginModal();
    }
}

function showLoginModal() {
    document.getElementById('login-modal').style.display = 'flex';
    document.getElementById('app-container').style.display = 'none';
}

function showApp() {
    document.getElementById('login-modal').style.display = 'none';
    document.getElementById('app-container').style.display = 'flex';
    
    // Update user info in header
    document.getElementById('user-name').textContent = currentUser.name;
    if (currentUser.picture) {
        const avatar = document.getElementById('user-avatar');
        avatar.src = currentUser.picture;
        avatar.style.display = 'block';
    }
}

async function tryRefreshToken() {
    if (!refreshToken || isRefreshing) {
        return false;
    }
    
    isRefreshing = true;
    
    try {
        const response = await fetch('/api/v1/auth/refresh', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                refresh_token: refreshToken
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            authToken = data.access_token;
            localStorage.setItem('access_token', authToken);
            console.log('Token refreshed successfully');
            return true;
        } else {
            const errorData = await response.json();
            console.error('Token refresh failed:', errorData.detail);
            return false;
        }
    } catch (error) {
        console.error('Token refresh error:', error);
        return false;
    } finally {
        isRefreshing = false;
    }
}

function clearTokens() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    authToken = null;
    refreshToken = null;
    currentUser = null;
}

// Logout handler
document.getElementById('logout-btn')?.addEventListener('click', async () => {
    try {
        await fetch('/api/v1/auth/logout', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });
    } catch (error) {
        console.error('Logout error:', error);
    }
    
    clearTokens();
    showLoginModal();
});

// Helper function to make authenticated requests with auto-refresh
async function authenticatedFetch(url, options = {}) {
    if (!authToken) {
        throw new Error('Not authenticated');
    }
    
    options.headers = {
        ...options.headers,
        'Authorization': `Bearer ${authToken}`
    };
    
    let response = await fetch(url, options);
    
    if (response.status === 401 && !isRefreshing) {
        // Access token expired - try to refresh
        const refreshed = await tryRefreshToken();
        
        if (refreshed) {
            // Retry the original request with new token
            options.headers['Authorization'] = `Bearer ${authToken}`;
            response = await fetch(url, options);
        } else {
            // Refresh failed - force re-login
            clearTokens();
            showLoginModal();
            throw new Error('Session expired. Please login again.');
        }
    }
    
    return response;
}

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active states
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Query Form Handler
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const question = document.getElementById('question').value.trim();
    const enableStreaming = document.getElementById('enable-streaming').checked;
    const submitBtn = document.getElementById('query-submit');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const resultsDiv = document.getElementById('results');
    
    if (!question) return;
    
    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Processing...';
    loader.style.display = 'inline-block';
    resultsDiv.style.display = 'none';
    document.getElementById('suggested-questions').style.display = 'none';
    
    try {
        if (enableStreaming) {
            // Use streaming endpoint
            await streamQuery(question);
        } else {
            // Use traditional polling endpoint
            const response = await authenticatedFetch('/api/v1/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: question,
                    chat_history: chatHistory
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            const taskId = data.task_id;
            
            // Show results card
            resultsDiv.style.display = 'block';
            document.getElementById('status-badge').textContent = 'Processing';
            document.getElementById('status-badge').className = 'status-badge processing';
            document.getElementById('answer-content').innerHTML = '<p>Retrieving and analyzing papers...</p>';
            document.getElementById('citations-section').style.display = 'none';
            
            // Poll for results
            await pollQueryResults(taskId, question);
        }
        
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.style.display = 'block';
        document.getElementById('status-badge').textContent = 'Failed';
        document.getElementById('status-badge').className = 'status-badge failed';
        document.getElementById('answer-content').innerHTML = `<p style="color: var(--error-color);">Error: ${error.message}</p>`;
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Submit Query';
        loader.style.display = 'none';
    }
});

async function streamQuery(question) {
    const resultsDiv = document.getElementById('results');
    const statusBadge = document.getElementById('status-badge');
    const answerContent = document.getElementById('answer-content');
    const citationsSection = document.getElementById('citations-section');
    const citationsList = document.getElementById('citations-list');
    
    // Show results card
    resultsDiv.style.display = 'block';
    statusBadge.textContent = 'Streaming';
    statusBadge.className = 'status-badge processing';
    answerContent.innerHTML = '';
    citationsSection.style.display = 'none';
    
    let fullAnswer = '';
    let citations = [];
    
    try {
        const response = await fetch('/api/v1/query/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${authToken}`
            },
            body: JSON.stringify({ 
                query: question,
                chat_history: chatHistory
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;
                    
                    try {
                        const event = JSON.parse(data);
                        
                        if (event.type === 'start') {
                            answerContent.innerHTML = '<p class="streaming-indicator">Generating answer...</p>';
                        } else if (event.type === 'sources') {
                            statusBadge.textContent = 'Generating';
                        } else if (event.type === 'answer_start') {
                            answerContent.innerHTML = '';
                        } else if (event.type === 'chunk') {
                            fullAnswer += event.content;
                            answerContent.innerHTML = formatAnswer(fullAnswer) + '<span class="cursor">|</span>';
                        } else if (event.type === 'citations') {
                            citations = event.data;
                        } else if (event.type === 'done') {
                            statusBadge.textContent = 'Completed';
                            statusBadge.className = 'status-badge completed';
                            answerContent.innerHTML = formatAnswer(fullAnswer);
                            
                            // Display citations
                            if (citations.length > 0) {
                                citationsSection.style.display = 'block';
                                citationsList.innerHTML = citations.map(citation => `
                                    <div class="citation-item">
                                        <div style="display: flex; align-items: start;">
                                            <span class="citation-number">${citation.number}</span>
                                            <div style="flex: 1;">
                                                <div class="citation-title">${citation.title}</div>
                                                <div class="citation-meta">
                                                    arXiv: <a href="${citation.url}" target="_blank" class="citation-link">${citation.arxiv_id}</a>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                `).join('');
                            }
                            
                            // Add to chat history
                            addToChatHistory(question, fullAnswer, citations);
                            
                            // Generate suggested questions
                            await generateSuggestedQuestions(question, fullAnswer);
                        } else if (event.type === 'error') {
                            throw new Error(event.message);
                        }
                    } catch (e) {
                        // Ignore JSON parse errors for incomplete chunks
                        if (!(e instanceof SyntaxError)) {
                            throw e;
                        }
                    }
                }
            }
        }
    } catch (error) {
        statusBadge.textContent = 'Failed';
        statusBadge.className = 'status-badge failed';
        answerContent.innerHTML = `<p style="color: var(--error-color);">Error: ${error.message}</p>`;
        throw error;
    }
}

async function pollQueryResults(taskId, question) {
    const maxAttempts = 60;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        try {
            const response = await authenticatedFetch(`/api/v1/query/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                displayQueryResults(data.result, question);
                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Query processing failed');
            }
            
            attempts++;
        } catch (error) {
            throw error;
        }
    }
    
    throw new Error('Query timeout - please try again');
}

function displayQueryResults(result, question) {
    const statusBadge = document.getElementById('status-badge');
    const answerContent = document.getElementById('answer-content');
    const citationsSection = document.getElementById('citations-section');
    const citationsList = document.getElementById('citations-list');
    
    statusBadge.textContent = 'Completed';
    statusBadge.className = 'status-badge completed';
    
    // Display answer with proper formatting
    const formattedAnswer = formatAnswer(result.answer);
    answerContent.innerHTML = formattedAnswer;
    
    // Display citations
    if (result.citations && result.citations.length > 0) {
        citationsSection.style.display = 'block';
        citationsList.innerHTML = result.citations.map(citation => `
            <div class="citation-item">
                <div style="display: flex; align-items: start;">
                    <span class="citation-number">${citation.number}</span>
                    <div style="flex: 1;">
                        <div class="citation-title">${citation.title}</div>
                        <div class="citation-meta">
                            arXiv: <a href="${citation.url}" target="_blank" class="citation-link">${citation.arxiv_id}</a>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    // Add to chat history
    addToChatHistory(question, result.answer, result.citations);
    
    // Generate suggested questions
    generateSuggestedQuestions(question, result.answer);
}

function addToChatHistory(query, answer, citations) {
    chatHistory.push({
        query: query,
        answer: answer,
        citations: citations,
        timestamp: new Date().toISOString()
    });
    
    // Keep only last 10 exchanges
    if (chatHistory.length > 10) {
        chatHistory = chatHistory.slice(-10);
    }
    
    // Save to localStorage
    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
    
    // Update display
    displayChatHistory();
}

function displayChatHistory() {
    const historyDiv = document.getElementById('chat-history');
    const historyList = document.getElementById('chat-history-list');
    
    if (chatHistory.length === 0) {
        historyDiv.style.display = 'none';
        return;
    }
    
    historyDiv.style.display = 'block';
    historyList.innerHTML = chatHistory.slice().reverse().map((item, index) => `
        <div class="history-item" data-index="${chatHistory.length - 1 - index}">
            <div class="history-query">
                <strong>Q:</strong> ${escapeHtml(item.query)}
            </div>
            <div class="history-answer">
                <strong>A:</strong> ${escapeHtml(item.answer.substring(0, 200))}${item.answer.length > 200 ? '...' : ''}
            </div>
            <div class="history-meta">
                ${new Date(item.timestamp).toLocaleString()}
                <button class="btn-link" onclick="reaskQuestion('${escapeHtml(item.query)}')">Ask Again</button>
            </div>
        </div>
    `).join('');
}

function reaskQuestion(query) {
    document.getElementById('question').value = query;
    document.getElementById('question').scrollIntoView({ behavior: 'smooth' });
}

async function generateSuggestedQuestions(query, answer) {
    try {
        const response = await authenticatedFetch('/api/v1/suggest-questions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_query: query,
                answer: answer,
                chat_history: chatHistory
            })
        });
        
        if (!response.ok) {
            console.error('Failed to generate suggestions');
            return;
        }
        
        const data = await response.json();
        const taskId = data.task_id;
        
        // Poll for suggestions
        await pollSuggestedQuestions(taskId);
        
    } catch (error) {
        console.error('Error generating suggestions:', error);
    }
}

async function pollSuggestedQuestions(taskId) {
    const maxAttempts = 20;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        try {
            const response = await authenticatedFetch(`/api/v1/ingest/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                displaySuggestedQuestions(data.result.suggestions);
                return;
            } else if (data.status === 'failed') {
                console.error('Suggestion generation failed');
                return;
            }
            
            attempts++;
        } catch (error) {
            console.error('Error polling suggestions:', error);
            return;
        }
    }
}

function displaySuggestedQuestions(suggestions) {
    if (!suggestions || suggestions.length === 0) return;
    
    const suggestionsDiv = document.getElementById('suggested-questions');
    const suggestionsList = document.getElementById('suggestions-list');
    
    suggestionsDiv.style.display = 'block';
    suggestionsList.innerHTML = suggestions.map((question, index) => `
        <button class="suggestion-btn" onclick="askSuggestedQuestion('${escapeHtml(question)}')">
            ${escapeHtml(question)}
        </button>
    `).join('');
}

function askSuggestedQuestion(question) {
    // Unescape the question
    const textarea = document.createElement('textarea');
    textarea.innerHTML = question;
    const unescapedQuestion = textarea.value;
    
    document.getElementById('question').value = unescapedQuestion;
    document.getElementById('query-form').dispatchEvent(new Event('submit'));
}

function formatAnswer(answer) {
    // Split into paragraphs and format
    const paragraphs = answer.split('\n\n').filter(p => p.trim());
    return paragraphs.map(p => `<p>${escapeHtml(p)}</p>`).join('');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Ingest Form Handler
document.getElementById('ingest-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const category = document.getElementById('category').value.trim();
    const maxResults = parseInt(document.getElementById('max-results').value);
    const skipPapers = parseInt(document.getElementById('skip-papers').value);
    const submitBtn = document.getElementById('ingest-submit');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const resultsDiv = document.getElementById('ingest-results');
    
    if (!category) return;
    
    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Starting...';
    loader.style.display = 'inline-block';
    resultsDiv.style.display = 'none';
    
    try {
        // Submit ingestion
        const response = await authenticatedFetch('/api/v1/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: category, 
                max_results: maxResults,
                skip_papers: skipPapers
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        const taskId = data.task_id;
        
        // Show results card
        resultsDiv.style.display = 'block';
        document.getElementById('ingest-status-badge').textContent = 'Processing';
        document.getElementById('ingest-status-badge').className = 'status-badge processing';
        document.getElementById('ingest-content').innerHTML = `
            <div class="status-item">
                <span class="status-label">Status:</span>
                <span class="status-value">Downloading and processing papers...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Category:</span>
                <span class="status-value">${category}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Max Papers:</span>
                <span class="status-value">${maxResults}</span>
            </div>
            ${skipPapers > 0 ? `<div class="status-item">
                <span class="status-label">Skipping First:</span>
                <span class="status-value">${skipPapers} papers</span>
            </div>` : ''}
        `;
        
        // Poll for results
        await pollIngestResults(taskId, category, maxResults, skipPapers);
        
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.style.display = 'block';
        document.getElementById('ingest-status-badge').textContent = 'Failed';
        document.getElementById('ingest-status-badge').className = 'status-badge failed';
        document.getElementById('ingest-content').innerHTML = `
            <p style="color: var(--error-color);">Error: ${error.message}</p>
        `;
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Start Ingestion';
        loader.style.display = 'none';
    }
});

async function pollIngestResults(taskId, category, maxResults, skipPapers) {
    const maxAttempts = 120;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        try {
            const response = await authenticatedFetch(`/api/v1/ingest/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                displayIngestResults(data.result, category, maxResults, skipPapers);
                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Ingestion failed');
            }
            
            attempts++;
        } catch (error) {
            throw error;
        }
    }
    
    throw new Error('Ingestion timeout - please check logs');
}

function displayIngestResults(result, category, maxResults, skipPapers) {
    const statusBadge = document.getElementById('ingest-status-badge');
    const ingestContent = document.getElementById('ingest-content');
    
    const isSuccess = result.status === 'success';
    
    statusBadge.textContent = isSuccess ? 'Completed' : 'Completed with warnings';
    statusBadge.className = `status-badge ${isSuccess ? 'completed' : 'warning'}`;
    
    ingestContent.innerHTML = `
        <div class="status-item">
            <span class="status-label">Status:</span>
            <span class="status-value">${result.status}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Category:</span>
            <span class="status-value">${category}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Papers Ingested:</span>
            <span class="status-value">${result.papers_ingested || 0}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Chunks Added:</span>
            <span class="status-value">${result.chunks_added || 0}</span>
        </div>
        ${skipPapers > 0 ? `<div class="status-item">
            <span class="status-label">Papers Skipped:</span>
            <span class="status-value">${skipPapers}</span>
        </div>` : ''}
        <div style="margin-top: 16px; padding: 12px; background: var(--surface); border-radius: 8px;">
            <p style="color: var(--success-color); font-weight: 500;">
                Papers successfully added to the database and ready for querying.
            </p>
        </div>
    `;
}

// Upload Form Handler
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('pdf-file');
    const file = fileInput.files[0];
    const submitBtn = document.getElementById('upload-submit');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const resultsDiv = document.getElementById('ingest-results');
    
    if (!file) {
        alert('Please select a PDF file');
        return;
    }
    
    // Validate file type
    if (!file.name.endsWith('.pdf')) {
        alert('Only PDF files are supported');
        return;
    }
    
    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Uploading...';
    loader.style.display = 'inline-block';
    resultsDiv.style.display = 'none';
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        // Upload file
        const response = await fetch('/api/v1/upload', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`
            },
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        const taskId = data.task_id;
        
        // Show results card
        resultsDiv.style.display = 'block';
        document.getElementById('ingest-status-badge').textContent = 'Processing';
        document.getElementById('ingest-status-badge').className = 'status-badge processing';
        document.getElementById('ingest-content').innerHTML = `
            <div class="status-item">
                <span class="status-label">Status:</span>
                <span class="status-value">Processing uploaded PDF...</span>
            </div>
            <div class="status-item">
                <span class="status-label">Filename:</span>
                <span class="status-value">${file.name}</span>
            </div>
        `;
        
        // Poll for results
        await pollUploadResults(taskId, file.name);
        
        // Clear file input
        fileInput.value = '';
        
    } catch (error) {
        console.error('Error:', error);
        resultsDiv.style.display = 'block';
        document.getElementById('ingest-status-badge').textContent = 'Failed';
        document.getElementById('ingest-status-badge').className = 'status-badge failed';
        document.getElementById('ingest-content').innerHTML = `
            <p style="color: var(--error-color);">Error: ${error.message}</p>
        `;
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'Upload Paper';
        loader.style.display = 'none';
    }
});

async function pollUploadResults(taskId, filename) {
    const maxAttempts = 60;
    let attempts = 0;
    
    while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        try {
            const response = await authenticatedFetch(`/api/v1/ingest/${taskId}`);
            const data = await response.json();
            
            if (data.status === 'completed') {
                displayUploadResults(data.result, filename);
                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Upload processing failed');
            }
            
            attempts++;
        } catch (error) {
            throw error;
        }
    }
    
    throw new Error('Upload processing timeout');
}

function displayUploadResults(result, filename) {
    const statusBadge = document.getElementById('ingest-status-badge');
    const ingestContent = document.getElementById('ingest-content');
    
    const isSuccess = result.status === 'success';
    
    statusBadge.textContent = isSuccess ? 'Completed' : 'Completed with warnings';
    statusBadge.className = `status-badge ${isSuccess ? 'completed' : 'warning'}`;
    
    ingestContent.innerHTML = `
        <div class="status-item">
            <span class="status-label">Status:</span>
            <span class="status-value">${result.status}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Filename:</span>
            <span class="status-value">${filename}</span>
        </div>
        <div class="status-item">
            <span class="status-label">Chunks Added:</span>
            <span class="status-value">${result.chunks_added || 0}</span>
        </div>
        <div style="margin-top: 16px; padding: 12px; background: var(--surface); border-radius: 8px;">
            <p style="color: var(--success-color); font-weight: 500;">
                ${result.message || 'Paper successfully processed and ready for querying.'}
            </p>
        </div>
    `;
}
