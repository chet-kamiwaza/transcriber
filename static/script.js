// Client‑side logic for the Whisper web transcription app.
//
// This script manages saving the API key, uploading an audio file, displaying
// progress during the upload/transcription process, and rendering the resulting
// transcript. It uses XMLHttpRequest instead of the Fetch API to provide
// fine‑grained upload progress events.

document.addEventListener('DOMContentLoaded', function () {
    const apiKeySection = document.getElementById('apiKeySection');
    const uploadSection = document.getElementById('uploadSection');
    const saveKeyBtn = document.getElementById('saveKeyBtn');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const audioInput = document.getElementById('audioInput');
    const transcribeBtn = document.getElementById('transcribeBtn');
    const languageInput = document.getElementById('languageInput');
    const promptInput = document.getElementById('promptInput');
    const statusP = document.getElementById('status');
    const progressBar = document.getElementById('progressBar');
    const progressBarInner = progressBar.querySelector('div');
    const transcriptOutput = document.getElementById('transcriptOutput');
    const downloadBtn = document.getElementById('downloadBtn');

    // Show/hide sections based on whether the server indicated an API key exists
    if (window.hasApiKey) {
        apiKeySection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
    }

    // Save the API key
    saveKeyBtn.addEventListener('click', function () {
        const key = apiKeyInput.value.trim();
        if (!key) {
            alert('Please enter a valid API key.');
            return;
        }
        fetch('/set_api_key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key })
        })
            .then((res) => res.json())
            .then((data) => {
                if (data.status) {
                    apiKeySection.classList.add('hidden');
                    uploadSection.classList.remove('hidden');
                } else {
                    alert(data.error || 'Failed to save API key.');
                }
            })
            .catch((err) => {
                console.error(err);
                alert('Error saving API key.');
            });
    });

    // Transcribe the selected file
    transcribeBtn.addEventListener('click', function () {
        const file = audioInput.files[0];
        if (!file) {
            alert('Please select an audio file to transcribe.');
            return;
        }
        const language = languageInput.value.trim();
        const prompt = promptInput.value.trim();
        const formData = new FormData();
        formData.append('audio', file);
        // Always use gpt-4o-transcribe on the server, but include the
        // language and prompt parameters for the API call.  The server
        // ignores any model field supplied by the client.
        if (language) {
            formData.append('language', language);
        }
        if (prompt) {
            formData.append('prompt', prompt);
        }

        // Reset progress UI
        progressBar.classList.remove('hidden');
        progressBarInner.style.width = '0%';
        statusP.textContent = 'Uploading…';
        transcriptOutput.value = '';
        downloadBtn.classList.add('hidden');

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/transcribe', true);
        // Track upload progress
        if (xhr.upload) {
            xhr.upload.onprogress = function (event) {
                if (event.lengthComputable) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    progressBarInner.style.width = percentComplete.toFixed(0) + '%';
                }
            };
        }
        // Show an animated status while the server processes the file after upload
        let dots = 0;
        let processingInterval;
        xhr.upload.onloadend = function () {
            // Upload finished; switch status to processing and keep the bar full
            progressBarInner.style.width = '100%';
            statusP.textContent = 'Processing…';
            dots = 0;
            // Animate dots to indicate activity
            processingInterval = setInterval(() => {
                dots = (dots + 1) % 4;
                statusP.textContent = 'Processing' + '.'.repeat(dots);
            }, 700);
        };
        xhr.onreadystatechange = function () {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                progressBarInner.style.width = '100%';
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (xhr.status === 200) {
                        transcriptOutput.value = response.transcript || '';
                        statusP.textContent = 'Completed.';
                        // Stop animated status when done
                        if (processingInterval) clearInterval(processingInterval);
                        if (response.transcript) {
                            downloadBtn.classList.remove('hidden');
                        }
                    } else {
                        statusP.textContent = 'Error: ' + (response.error || 'Transcription failed.');
                        if (processingInterval) clearInterval(processingInterval);
                    }
                } catch (err) {
                    console.error(err);
                    statusP.textContent = 'Error: failed to parse server response.';
                    if (processingInterval) clearInterval(processingInterval);
                }
            }
        };
        xhr.send(formData);
    });

    // Download the transcript as a text file
    downloadBtn.addEventListener('click', function () {
        const blob = new Blob([transcriptOutput.value], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'transcript.txt';
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
    });
});