const MAX_SEQUENCE = 40;
const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];
const MODEL_HEALTH_URL = "/api/health";
const MODEL_PREDICT_URL = "/api/predict";

const inputVideo = document.getElementById("input-video");
const outputCanvas = document.getElementById("output-canvas");
const cameraState = document.getElementById("camera-state");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsList = document.getElementById("diagnostics-list");
const modelStatusTitle = document.getElementById("model-status-title");
const modelHealthBadge = document.getElementById("model-health-badge");
const topPredictionLabel = document.getElementById("top-prediction-label");
const topPredictionScore = document.getElementById("top-prediction-score");
const bufferedFrames = document.getElementById("buffered-frames");
const lastInferenceAt = document.getElementById("last-inference-at");
const predictionList = document.getElementById("prediction-list");
const retryModelBtn = document.getElementById("retry-model-btn");
const retryCameraBtn = document.getElementById("retry-camera-btn");
const resetBufferBtn = document.getElementById("reset-buffer-btn");

const canvasCtx = outputCanvas.getContext("2d");

let camera = null;
let holistic = null;
let frameBuffer = [];
let predictionInFlight = false;
let frameCounter = 0;
let lastModelInfo = null;
let cameraReady = false;

function formatTime(date) {
    return new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
}

function renderDiagnosticRows(rows) {
    diagnosticsList.innerHTML = "";
    for (const rowData of rows) {
        const row = document.createElement("div");
        row.className = "gesture-diagnostic-row";
        const copy = document.createElement("div");
        copy.className = "gesture-diagnostic-copy";
        const label = document.createElement("div");
        label.className = "gesture-diagnostic-label";
        label.textContent = rowData.label;
        const value = document.createElement("div");
        value.className = "gesture-diagnostic-value";
        value.textContent = rowData.value;
        const badge = document.createElement("span");
        badge.className = `gesture-check-badge ${rowData.tone || "is-neutral"}`;
        badge.textContent = rowData.badge;
        copy.append(label, value);
        row.append(copy, badge);
        diagnosticsList.appendChild(row);
    }
}

function updateDiagnostics(extra = {}) {
    diagnosticsSummary.textContent = extra.summary || "Capture is running locally through the browser and localhost inference API.";
    renderDiagnosticRows([
        {
            label: "Model API",
            value: lastModelInfo ? `Loaded ${lastModelInfo.num_classes} classes from ${lastModelInfo.model_name}.` : "Health check has not completed yet.",
            badge: lastModelInfo ? "Ready" : "Checking",
            tone: lastModelInfo ? "is-good" : "is-neutral"
        },
        {
            label: "Camera state",
            value: cameraReady ? "Video frames are available for holistic tracking." : "Camera stream is not ready yet.",
            badge: cameraReady ? "Live" : "Waiting",
            tone: cameraReady ? "is-good" : "is-neutral"
        },
        {
            label: "Frame buffer",
            value: `${frameBuffer.length} / ${MAX_SEQUENCE} frames collected for the next inference window.`,
            badge: `${frameBuffer.length}`,
            tone: frameBuffer.length >= MAX_SEQUENCE ? "is-good" : "is-neutral"
        },
        {
            label: "Inference loop",
            value: predictionInFlight ? "A prediction request is currently running." : "The browser is ready to send the next sequence.",
            badge: predictionInFlight ? "Busy" : "Idle",
            tone: predictionInFlight ? "is-neutral" : "is-good"
        }
    ]);
}

function emptyHand() {
    return Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
}

function emptyPose() {
    return Array.from({ length: POSE_IDS.length }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
}

function toLandmarkArray(landmarks, count) {
    if (!landmarks?.length) {
        return Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0 }));
    }
    return Array.from({ length: count }, (_, index) => landmarks[index] || { x: 0, y: 0, z: 0 });
}

function toPoseSubset(landmarks) {
    if (!landmarks?.length) {
        return emptyPose();
    }
    return POSE_IDS.map(index => landmarks[index] || { x: 0, y: 0, z: 0, visibility: 0 });
}

function normalizeLandmarks(leftHand, rightHand, pose) {
    const leftShoulder = pose[1];
    const rightShoulder = pose[2];
    const shoulderVisible = leftShoulder.visibility >= 0.3 && rightShoulder.visibility >= 0.3;

    let centerX = 0.5;
    let centerY = 0.5;
    let scale = 0.15;

    if (shoulderVisible) {
        centerX = (leftShoulder.x + rightShoulder.x) / 2;
        centerY = (leftShoulder.y + rightShoulder.y) / 2;
        scale = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
    } else {
        const wrists = [];
        if (leftHand.some(point => point.x || point.y || point.z)) {
            wrists.push(leftHand[0]);
        }
        if (rightHand.some(point => point.x || point.y || point.z)) {
            wrists.push(rightHand[0]);
        }
        if (wrists.length) {
            centerX = wrists.reduce((sum, point) => sum + point.x, 0) / wrists.length;
            centerY = wrists.reduce((sum, point) => sum + point.y, 0) / wrists.length;
        }
    }

    scale = Math.max(scale, 1e-4);

    const normalize3 = points => points.flatMap(point => [
        (point.x - centerX) / scale,
        (point.y - centerY) / scale,
        point.z / scale
    ]);

    const normalizePose = points => points.flatMap(point => [
        (point.x - centerX) / scale,
        (point.y - centerY) / scale,
        point.z / scale,
        point.visibility || 0
    ]);

    return [
        ...normalize3(leftHand),
        ...normalize3(rightHand),
        ...normalizePose(pose)
    ];
}

function appendFrame(results) {
    const leftHand = toLandmarkArray(results.leftHandLandmarks, 21);
    const rightHand = toLandmarkArray(results.rightHandLandmarks, 21);
    const pose = toPoseSubset(results.poseLandmarks);
    const featureVector = normalizeLandmarks(leftHand, rightHand, pose);
    frameBuffer.push(featureVector);
    if (frameBuffer.length > MAX_SEQUENCE) {
        frameBuffer.shift();
    }
    bufferedFrames.textContent = `${frameBuffer.length}`;
    updateDiagnostics();
}

function drawResults(results) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    const drawLandmarkSet = (landmarks, connections, connectorColor, fillColor) => {
        if (!landmarks) { return; }
        window.drawConnectors(canvasCtx, landmarks, connections, { color: connectorColor, lineWidth: 3 });
        window.drawLandmarks(canvasCtx, landmarks, { color: "#eff6ff", fillColor, radius: 3 });
    };

    drawLandmarkSet(results.poseLandmarks, window.POSE_CONNECTIONS, "#fb7185", "#be123c");
    drawLandmarkSet(results.leftHandLandmarks, window.HAND_CONNECTIONS, "#f97316", "#7c2d12");
    drawLandmarkSet(results.rightHandLandmarks, window.HAND_CONNECTIONS, "#38bdf8", "#1d4ed8");
    canvasCtx.restore();
}

function renderPredictions(predictions) {
    predictionList.innerHTML = "";
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${item.label}`;
        const value = document.createElement("span");
        value.textContent = `${Math.round(item.score * 100)}%`;
        value.className = `gesture-check-badge ${index === 0 ? "is-good" : "is-neutral"}`;
        row.append(label, value);
        predictionList.appendChild(row);
    });
}

async function predictSequence() {
    if (predictionInFlight || frameBuffer.length < MAX_SEQUENCE || !lastModelInfo) {
        return;
    }
    predictionInFlight = true;
    updateDiagnostics();
    try {
        const response = await fetch(MODEL_PREDICT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sequence: frameBuffer })
        });
        if (!response.ok) {
            throw new Error(`Prediction failed with status ${response.status}`);
        }
        const result = await response.json();
        const best = result.predictions[0];
        topPredictionLabel.textContent = best ? best.label : "No prediction";
        topPredictionScore.textContent = best ? `Confidence ${Math.round(best.score * 100)}%` : "No confidence available";
        renderPredictions(result.predictions);
        lastInferenceAt.textContent = formatTime(new Date());
    } catch (error) {
        console.error(error);
        topPredictionLabel.textContent = "Prediction failed";
        topPredictionScore.textContent = error.message;
    } finally {
        predictionInFlight = false;
        updateDiagnostics();
    }
}

async function checkModel() {
    modelStatusTitle.textContent = "Checking local model...";
    modelHealthBadge.textContent = "Checking";
    try {
        const response = await fetch(MODEL_HEALTH_URL);
        if (!response.ok) {
            throw new Error(`Health check failed with status ${response.status}`);
        }
        lastModelInfo = await response.json();
        modelStatusTitle.textContent = "Local model is ready";
        modelHealthBadge.textContent = `${lastModelInfo.num_classes} classes`;
        updateDiagnostics();
    } catch (error) {
        console.error(error);
        lastModelInfo = null;
        modelStatusTitle.textContent = "Model API unavailable";
        modelHealthBadge.textContent = "Offline";
        diagnosticsSummary.textContent = `Could not reach localhost model server: ${error.message}`;
        updateDiagnostics({ summary: `Could not reach localhost model server: ${error.message}` });
    }
}

async function startCamera() {
    cameraState.textContent = "Starting camera...";
    cameraReady = false;
    updateDiagnostics();
    if (camera) {
        camera.stop();
        camera = null;
    }
    if (!holistic) {
        holistic = new window.Holistic({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        holistic.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        holistic.onResults(async results => {
            drawResults(results);
            appendFrame(results);
            frameCounter += 1;
            cameraReady = true;
            cameraState.textContent = "Camera is live";
            if (frameCounter % 6 === 0) {
                await predictSequence();
            }
        });
    }

    camera = new window.Camera(inputVideo, {
        onFrame: async () => {
            await holistic.send({ image: inputVideo });
        },
        width: 1280,
        height: 720
    });

    try {
        await camera.start();
    } catch (error) {
        console.error(error);
        cameraState.textContent = "Camera failed";
        diagnosticsSummary.textContent = `Camera could not start: ${error.message}`;
    }
    updateDiagnostics();
}

retryModelBtn.addEventListener("click", () => {
    checkModel().catch(console.error);
});

retryCameraBtn.addEventListener("click", () => {
    startCamera().catch(console.error);
});

resetBufferBtn.addEventListener("click", () => {
    frameBuffer = [];
    bufferedFrames.textContent = "0";
    topPredictionLabel.textContent = "Waiting for frames";
    topPredictionScore.textContent = "Need 40 frames before the first inference.";
    predictionList.innerHTML = "";
    updateDiagnostics({ summary: "The sequence buffer was cleared. Hold a sign steady to collect a new 40-frame window." });
});

checkModel().catch(console.error);
startCamera().catch(console.error);
