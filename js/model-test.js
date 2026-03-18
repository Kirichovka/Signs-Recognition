import {
    MAX_SEQUENCE,
    drawHolisticResults,
    featureVectorFromResults,
    getCameraPermissionState,
    loadBrowserModel,
    predictWithBrowserModel,
    prettifyLabel,
    startCameraStream,
    stopMediaStream,
    describeCameraError
} from "./sign-model-runtime.js?v=20260318-7";

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

let activeStream = null;
let animationFrameId = 0;
let holistic = null;
let frameBuffer = [];
let predictionInFlight = false;
let frameCounter = 0;
let modelState = null;
let cameraReady = false;

function isImageModel() {
    return modelState?.model_type === "image";
}

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

async function updateDiagnostics(extra = {}) {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = extra.summary || "Everything is running directly in the browser. No Python backend is required for this page.";
    renderDiagnosticRows([
        {
            label: "Browser model",
            value: modelState ? `Loaded ${modelState.num_classes} labels from ${modelState.model_name}.` : "Model has not loaded yet.",
            badge: modelState ? "Ready" : "Loading",
            tone: modelState ? "is-good" : "is-neutral"
        },
        {
            label: "Camera state",
            value: cameraReady ? "Video frames are available for holistic tracking." : "Camera stream is not ready yet.",
            badge: cameraReady ? "Live" : "Waiting",
            tone: cameraReady ? "is-good" : "is-neutral"
        },
        {
            label: "Permission",
            value: `Camera permission state: ${permissionState}.`,
            badge: permissionState,
            tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral"
        },
        {
            label: "Frame buffer",
            value: isImageModel()
                ? "Alphabet mode predicts from the current live frame."
                : `${frameBuffer.length} / ${MAX_SEQUENCE} frames collected for the next inference window.`,
            badge: isImageModel() ? "Live" : `${frameBuffer.length}`,
            tone: isImageModel() || frameBuffer.length >= MAX_SEQUENCE ? "is-good" : "is-neutral"
        },
        {
            label: "Inference loop",
            value: predictionInFlight
                ? "A prediction request is currently running inside ONNX Runtime Web."
                : isImageModel()
                    ? "The browser is ready to classify the current frame."
                    : "The browser is ready to run the next sequence.",
            badge: predictionInFlight ? "Busy" : "Idle",
            tone: predictionInFlight ? "is-neutral" : "is-good"
        }
    ]);
}

function renderPredictions(predictions) {
    predictionList.innerHTML = "";
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${prettifyLabel(item.label)}`;
        const value = document.createElement("span");
        value.textContent = `${Math.round(item.score * 100)}%`;
        value.className = `gesture-check-badge ${index === 0 ? "is-good" : "is-neutral"}`;
        row.append(label, value);
        predictionList.appendChild(row);
    });
}

async function predictSequence() {
    const waitingForSequence = !isImageModel() && frameBuffer.length < MAX_SEQUENCE;
    if (predictionInFlight || !modelState || waitingForSequence) {
        return;
    }
    predictionInFlight = true;
    await updateDiagnostics();
    try {
        const modelInput = isImageModel() ? inputVideo : frameBuffer;
        const result = await predictWithBrowserModel(modelState, modelInput);
        const best = result.predictions[0];
        topPredictionLabel.textContent = best ? prettifyLabel(best.label) : "No prediction";
        topPredictionScore.textContent = best ? `Confidence ${Math.round(best.score * 100)}%` : "No confidence available";
        renderPredictions(result.predictions);
        lastInferenceAt.textContent = formatTime(new Date());
    } catch (error) {
        console.error(error);
        topPredictionLabel.textContent = "Prediction failed";
        topPredictionScore.textContent = error.message;
    } finally {
        predictionInFlight = false;
        await updateDiagnostics();
    }
}

async function checkModel() {
    modelStatusTitle.textContent = "Loading browser model...";
    modelHealthBadge.textContent = "Loading";
    try {
        modelState = await loadBrowserModel();
        modelStatusTitle.textContent = "Browser model is ready";
        modelHealthBadge.textContent = isImageModel()
            ? `${modelState.num_classes} letters`
            : `${modelState.num_classes} classes`;
        topPredictionScore.textContent = isImageModel()
            ? "Live alphabet classification is ready."
            : "Need 40 frames before the first inference.";
        await updateDiagnostics();
    } catch (error) {
        console.error(error);
        modelState = null;
        modelStatusTitle.textContent = "Model unavailable";
        modelHealthBadge.textContent = "Offline";
        await updateDiagnostics({ summary: `Could not load the browser model: ${error.message}` });
    }
}

function stopLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = 0;
    }
    stopMediaStream(activeStream);
    activeStream = null;
    cameraReady = false;
}

function processResults(results) {
    drawHolisticResults(canvasCtx, outputCanvas, inputVideo, results);
    if (!isImageModel()) {
        frameBuffer.push(featureVectorFromResults(results));
        if (frameBuffer.length > MAX_SEQUENCE) {
            frameBuffer.shift();
        }
    }
    bufferedFrames.textContent = isImageModel() ? "Live" : `${frameBuffer.length}`;
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % 6 === 0) {
        predictSequence().catch(console.error);
    }
}

async function startCamera() {
    cameraState.textContent = "Starting camera...";
    stopLoop();
    await updateDiagnostics();

    if (!holistic) {
        holistic = new globalThis.Holistic({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        holistic.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        holistic.onResults(processResults);
    }

    try {
        activeStream = await startCameraStream();
        inputVideo.srcObject = activeStream;
        await inputVideo.play();
        const processFrame = async () => {
            if (!activeStream || inputVideo.readyState < 2) {
                animationFrameId = requestAnimationFrame(processFrame);
                return;
            }
            await holistic.send({ image: inputVideo });
            animationFrameId = requestAnimationFrame(processFrame);
        };
        animationFrameId = requestAnimationFrame(processFrame);
    } catch (error) {
        console.error(error);
        cameraState.textContent = "Camera failed";
        await updateDiagnostics({ summary: `Camera could not start: ${describeCameraError(error)}` });
    }
}

retryModelBtn.addEventListener("click", () => {
    checkModel().catch(console.error);
});

retryCameraBtn.addEventListener("click", () => {
    startCamera().catch(console.error);
});

resetBufferBtn.addEventListener("click", async () => {
    frameBuffer = [];
    bufferedFrames.textContent = isImageModel() ? "Live" : "0";
    topPredictionLabel.textContent = isImageModel() ? "Waiting for camera" : "Waiting for frames";
    topPredictionScore.textContent = isImageModel()
        ? "Live alphabet prediction updates from the current frame."
        : "Need 40 frames before the first inference.";
    predictionList.innerHTML = "";
    await updateDiagnostics({
        summary: isImageModel()
            ? "Alphabet mode does not use a sequence buffer. The next live frame will be classified automatically."
            : "The sequence buffer was cleared. Hold a sign steady to collect a new 40-frame window."
    });
});

window.addEventListener("beforeunload", () => {
    stopLoop();
});

checkModel().catch(console.error);
startCamera().catch(console.error);
