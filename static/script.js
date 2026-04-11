let currentMode = "home";
let currentModel = "yolov8n";
let running = false;
let lastSpoken = "";
let captureInterval = null;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

// Initialize Camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { 
        video.srcObject = stream;
        console.log("✅ Camera initialized");
    })
    .catch(err => {
        alert("Camera access denied: " + err.message);
        console.error("Camera error:", err);
    });

// Attach functions to window so HTML buttons can find them
window.setMode = (mode) => {
    if (currentMode === mode) return; // Don't do anything if mode is already selected
    
    currentMode = mode;
    const modeTextElement = document.getElementById("modeText");
    if (modeTextElement) {
        modeTextElement.innerText = "MODE: " + mode.toUpperCase();
    }
    document.querySelectorAll(".mode").forEach(b => b.classList.remove("active"));
    document.getElementById(mode).classList.add("active");
    addToLog("🔄 Mode switched to: " + mode.toUpperCase());
    
    // Announce mode change
    speak("Be safe now, " + mode + " mode is on.");
};

window.setModel = (model) => {
    if (currentModel === model) return; // Don't do anything if model is already selected
    
    currentModel = model;
    addToLog("🔧 Model switched to: " + model);
};

window.start = () => {
    if (running) return; // Prevent multiple intervals
    running = true;
    addToLog("✓ System Started");
    addToLog("📊 Active - Model: " + currentModel + " | Mode: " + currentMode);
    
    // Clear any existing interval and create a new one
    if (captureInterval) clearInterval(captureInterval);
    captureInterval = setInterval(captureAndSend, 1500);
};

window.stop = () => {
    if (!running) return; // Already stopped
    running = false;
    
    // Stop the capture interval
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    // Clear the canvas
    if (ctx) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    }
    
    addToLog("✗ System Stopped");
};

function speak(text) {
    window.speechSynthesis.cancel(); // Stop previous speech
    const msg = new SpeechSynthesisUtterance(text);
    msg.rate = 0.9; // Slightly slower for clarity
    msg.pitch = 1.0;
    msg.volume = 1.0;
    window.speechSynthesis.speak(msg);
}

function addToLog(msg) {
    const log = document.getElementById('log');
    const entry = document.createElement('div');
    entry.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    log.prepend(entry);
}



function captureAndSend() {
    if (!running || !video.readyState || video.readyState !== video.HAVE_ENOUGH_DATA) return;

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;

    captureCtx.drawImage(video, 0, 0);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

    fetch('/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            image: dataUrl, 
            mode: currentMode,
            model: currentModel 
        })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                addToLog("⚠ Error: " + data.error);
                return;
            }
            
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            
            // ===== BACKGROUND COLOR ALERT FOR FACE DANGER =====
            // RED alert if known face within 200cm, GREEN if safe
            const faceDangerBgColor = data.has_face_danger ? "rgba(255, 0, 0, 0.15)" : "rgba(0, 255, 0, 0.05)";
            ctx.fillStyle = faceDangerBgColor;
            ctx.fillRect(0, 0, overlay.width, overlay.height);
            
            // ===== DRAW RECOGNIZED FACES =====
            if (data.face_detections && data.face_detections.length > 0) {
                data.face_detections.forEach(face => {
                    const [left, top, right, bottom] = face.box;
                    const boxWidth = right - left;
                    const boxHeight = bottom - top;
                    
                    // Draw box - red for danger (known+close), green for known+far, yellow for unknown
                    let boxColor = "#ffff00"; // yellow for unknown
                    if (face.is_known) {
                        boxColor = face.is_danger ? "#ff0000" : "#00ff00"; // red if close, green if far
                    }
                    ctx.strokeStyle = boxColor;
                    ctx.lineWidth = 3;
                    ctx.strokeRect(left, top, boxWidth, boxHeight);
                    
                    // Draw name background
                    let bgColor = "#aaaa00"; // yellow for unknown
                    if (face.is_known) {
                        bgColor = face.is_danger ? "#cc0000" : "#00aa00"; // red if close, green if far
                    }
                    ctx.fillStyle = bgColor;
                    ctx.fillRect(left, top - 30, boxWidth, 30);
                    
                    // Draw name text
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "bold 18px Arial";
                    ctx.textAlign = "center";
                    const nameText = face.name + " - " + face.distance_text;
                    ctx.fillText(nameText, left + boxWidth / 2, top - 8);
                });
            }
            
            // Draw visual detections (mode-filtered)
            if (data.visual_detections && data.visual_detections.length > 0) {
                data.visual_detections.forEach(d => {
                    const boxWidth = d.box[2] - d.box[0];
                    const boxHeight = d.box[3] - d.box[1];
                    
                    // Draw box - red for danger, lime for normal
                    ctx.strokeStyle = d.danger ? "red" : "lime";
                    ctx.lineWidth = 4;
                    ctx.strokeRect(d.box[0], d.box[1], boxWidth, boxHeight);
                    
                    // Draw label and distance background
                    ctx.fillStyle = d.danger ? "#ff0000" : "#00aa00";
                    const labelText = `${d.label} - ${d.distance_text}`;
                    ctx.font = "bold 16px Arial";
                    const textWidth = ctx.measureText(labelText).width;
                    ctx.fillRect(d.box[0], d.box[1] - 30, textWidth + 10, 30);
                    
                    // Draw label and distance text
                    ctx.fillStyle = "#ffffff";
                    ctx.textAlign = "left";
                    ctx.fillText(labelText, d.box[0] + 5, d.box[1] - 8);
                });
            }
            
            // ===== PRIORITY: SPEAK RECOGNIZED FACES IN DANGER ZONE FIRST =====
            let spokenSomething = false;
            
            // Check for known faces within danger zone (< 200cm)
            if (data.face_detections && data.face_detections.length > 0) {
                // First, check for faces in danger zone
                for (let face of data.face_detections) {
                    if (face.is_known && face.is_danger && face.confidence > 0.5) {
                        const sentence = `Alert! ${face.name} is approaching - ${face.distance_text}`;
                        if (sentence !== lastSpoken) {
                            addToLog(`🚨 ALERT: ${face.name} within 200cm (${face.distance_text})!`);
                            speak(sentence);
                            lastSpoken = sentence;
                            spokenSomething = true;
                            return; // Stop after alert
                        }
                    }
                }
                
                // Then check for other known faces outside danger zone
                if (!spokenSomething) {
                    for (let face of data.face_detections) {
                        if (face.is_known && !face.is_danger && face.confidence > 0.5) {
                            const sentence = `${face.name} is here at ${face.distance_text}`;
                            if (sentence !== lastSpoken) {
                                addToLog(`👤 Recognized: ${face.name} (${face.distance_text})`);
                                speak(sentence);
                                lastSpoken = sentence;
                                spokenSomething = true;
                                return; // Stop after recognizing
                            }
                        }
                    }
                }
            }
            
            // If no known faces in danger, speak obstacles
            if (!spokenSomething && data.audio_detections && data.audio_detections.length > 0) {
                // Sort by priority: danger first, then closest
                const sortedDetections = data.audio_detections.sort((a, b) => {
                    if (a.danger && !b.danger) return -1;
                    if (!a.danger && b.danger) return 1;
                    
                    // If same danger level, prioritize by closest distance
                    return a.distance_cm - b.distance_cm;
                });
                
                // Speak the most critical detection
                const d = sortedDetections[0];
                const confidence = Math.round(d.confidence * 100);
                let sentence = "";
                
                if (d.danger) {
                    sentence = `Danger! Obstacle ${d.distance_text} ${d.direction}`;
                } else {
                    sentence = `${d.label} ${d.distance_text} ${d.direction}`;
                }
                
                if (sentence !== lastSpoken) {
                    addToLog(`📊 Detected: ${sentence} | Confidence: ${confidence}%`);
                    speak(sentence);
                    lastSpoken = sentence;
                    spokenSomething = true;
                }
            }
            
            if (!spokenSomething) {
                // Show what we detected in the log
                let statusMsg = "👁️ Scanning...";
                if (data.face_detections && data.face_detections.length > 0) {
                    const knownCount = data.face_detections.filter(f => f.is_known).length;
                    const unknownCount = data.face_detections.length - knownCount;
                    if (knownCount > 0) statusMsg += ` | ${knownCount} known face(s)`;
                    if (unknownCount > 0) statusMsg += ` | ${unknownCount} unknown`;
                }
                addToLog(statusMsg);
            }
        })
        .catch(err => {
            addToLog("⚠ Connection error: " + err.message);
        });
}

// Initialize system log when page loads
window.addEventListener('load', () => {
    setTimeout(() => {
        addToLog("🚀 EYESONIC AI System Ready");
        addToLog("📊 Active Model: YOLOv8 Nano");
        addToLog("📍 Current Mode: HOME");
    }, 500);
});

window.trainFace = () => {
    if (!video.readyState || video.readyState !== video.HAVE_ENOUGH_DATA) {
        alert("Camera not ready. Please make sure the camera is working.");
        return;
    }

    const name = prompt("Enter the name of the person in front of the camera:");
    if (!name || name.trim() === "") {
        return; // Cancelled or null
    }
    
    addToLog(`📸 Capturing face to train as: ${name}...`);

    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;

    captureCtx.drawImage(video, 0, 0);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.9);

    fetch('/train_face', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            name: name,
            image: dataUrl
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            addToLog(data.message);
            alert("Success! " + data.message);
        } else {
            addToLog("⚠ Training Failed: " + data.message);
            alert("Error: " + data.message);
        }
    })
    .catch(err => {
        addToLog("⚠ Connection error during training: " + err.message);
        alert("Connection error: " + err.message);
    });
};