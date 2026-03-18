import {
    MAX_SEQUENCE,
    describeCameraError,
    drawHolisticResults,
    featureVectorFromResults,
    getCameraPermissionState,
    loadBrowserModel,
    predictWithBrowserModel,
    prettifyLabel,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.45;
const FALLBACK_LABELS = [
    "AXE1", "BACKPACK1", "BASKETBALL1", "BEE1", "BELIEVE1", "BELT1", "BITE1", "BREAKFAST1", "CALENDAR1", "CANCEL1",
    "CANCER1", "CHRISTMAS1", "CLOUD1", "CONFUSED1", "DARK1", "DEAF1", "DECIDE1", "DEMAND1", "DINNER1", "DOG1",
    "DOWNSIZE1", "DRAG1", "EAT1", "EDIT1", "ELEVATOR1", "FINE1", "FOREIGNER1", "GUESS1", "HALLOWEEN1", "HOSPITAL1",
    "HURDLE/TRIP1", "LETTUCE1", "LOCK1", "LUNCH1", "MECHANIC1", "MICROSCOPE1", "MOVIE1", "NIGHT1", "NOON1", "PARTY1",
    "PATIENT2", "RECENT1", "RESEARCH1", "RIVER1", "ROCKINGCHAIR1", "SHAVE1", "SPECIAL1", "THIRD1", "TYPE1", "WHATFOR1"
];

const gestureTitle = document.getElementById("gesture-title");
const gestureInstruction = document.getElementById("gesture-instruction");
const gestureEmoji = document.getElementById("gesture-emoji");
const gestureStep = document.getElementById("gesture-step");
const holdProgressBar = document.getElementById("hold-progress-bar");
const holdProgressValue = document.getElementById("hold-progress-value");
const gestureScore = document.getElementById("gesture-score");
const gestureStatus = document.getElementById("gesture-status");
const fingerChecklist = document.getElementById("finger-checklist");
const cameraState = document.getElementById("camera-state");
const inputVideo = document.getElementById("input-video");
const outputCanvas = document.getElementById("output-canvas");
const nextGestureBtn = document.getElementById("next-gesture-btn");
const prevGestureBtn = document.getElementById("prev-gesture-btn");
const retryCameraBtn = document.getElementById("retry-camera-btn");
const refreshDiagnosticsBtn = document.getElementById("refresh-diagnostics-btn");
const calibrateBtn = document.getElementById("calibrate-btn");
const calibrationStatus = document.getElementById("calibration-status");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsList = document.getElementById("diagnostics-list");

const canvasCtx = outputCanvas.getContext("2d");

let gestures = buildGestures(FALLBACK_LABELS);
let currentGestureIndex = 0;
let holdStartedAt = 0;
let lastSuccess = false;
let activeStream = null;
let animationFrameId = 0;
let holistic = null;
let frameBuffer = [];
let predictionInFlight = false;
let frameCounter = 0;
let lastCameraError = "";
let modelState = null;
let cameraReady = false;
let trackedHands = 0;

function buildGestures(labels) {
    return labels.map(label => {
        const title = prettifyLabel(label);
        const badgeBase = label.replace(/[0-9]+$/g, "").split(/[\/_]/)[0] || label;
        return {
            id: label,
            title,
            badge: badgeBase.slice(0, 8).toUpperCase(),
            instruction: `Show the sign for "${title}" and hold it steady until the progress bar fills.`
        };
    });
}

function getCurrentGesture() {
    return gestures[currentGestureIndex] || gestures[0];
}

function renderGestureCard() {
    const gesture = getCurrentGesture();
    gestureTitle.textContent = gesture.title;
    gestureInstruction.textContent = gesture.instruction;
    gestureEmoji.textContent = gesture.badge;
    gestureStep.textContent = `${currentGestureIndex + 1} / ${gestures.length}`;
}

function renderStatus(score, holdProgress, statusText) {
    const scoreValue = Math.round(score * 100);
    gestureScore.textContent = `${scoreValue}%`;
    gestureStatus.textContent = statusText;
    holdProgressBar.style.width = `${Math.round(holdProgress * 100)}%`;
    holdProgressValue.textContent = `${Math.round(holdProgress * 100)}%`;
    gestureScore.classList.toggle("is-strong", scoreValue >= 70);
}

function renderPredictions(predictions) {
    fingerChecklist.innerHTML = "";
    if (!predictions.length) {
        const empty = document.createElement("div");
        empty.className = "gesture-check-row";
        empty.innerHTML = '<span>Waiting</span><span class="gesture-check-badge is-neutral">No predictions yet</span>';
        fingerChecklist.appendChild(empty);
        return;
    }

    const target = getCurrentGesture().id;
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${prettifyLabel(item.label)}`;
        const value = document.createElement("span");
        value.textContent = `${Math.round(item.score * 100)}%`;
        const isTarget = item.label === target;
        value.className = `gesture-check-badge ${isTarget ? "is-good" : index === 0 ? "is-neutral" : "is-neutral"}`;
        row.append(label, value);
        fingerChecklist.appendChild(row);
    });
}

function setGesture(index) {
    currentGestureIndex = (index + gestures.length) % gestures.length;
    holdStartedAt = 0;
    lastSuccess = false;
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, "Show the target sign to the camera.");
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

async function collectDiagnostics(extra = {}) {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = extra.summary || "The trainer is running fully in the browser with ONNX Runtime Web.";
    renderDiagnosticRows([
        {
            label: "Browser model",
            value: modelState ? `Loaded ${modelState.num_classes} classes from ${modelState.model_name}.` : "Model has not loaded yet.",
            badge: modelState ? "Ready" : "Loading",
            tone: modelState ? "is-good" : "is-neutral"
        },
        {
            label: "Target sign",
            value: getCurrentGesture().title,
            badge: `${currentGestureIndex + 1}/${gestures.length}`,
            tone: "is-good"
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
            label: "Tracked hands",
            value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible in the frame.",
            badge: `${trackedHands}`,
            tone: trackedHands ? "is-good" : "is-neutral"
        },
        {
            label: "Frame buffer",
            value: `${frameBuffer.length} / ${MAX_SEQUENCE} frames buffered for inference.`,
            badge: `${frameBuffer.length}`,
            tone: frameBuffer.length >= MAX_SEQUENCE ? "is-good" : "is-neutral"
        },
        {
            label: "Last camera error",
            value: lastCameraError || "No camera error has been recorded in this session.",
            badge: lastCameraError ? "Has error" : "Clear",
            tone: lastCameraError ? "is-bad" : "is-good"
        }
    ]);
}

async function loadModel() {
    calibrationStatus.textContent = "Loading browser model...";
    try {
        modelState = await loadBrowserModel();
        gestures = buildGestures(modelState.label_names);
        currentGestureIndex = Math.min(currentGestureIndex, gestures.length - 1);
        renderGestureCard();
        calibrationStatus.textContent = `Browser model ready. Loaded ${modelState.num_classes} trained signs from ${modelState.model_name}.`;
    } catch (error) {
        console.error(error);
        modelState = null;
        calibrationStatus.textContent = `Could not load browser model: ${error.message}`;
    }
    await collectDiagnostics();
}

function evaluatePredictions(predictions) {
    const target = getCurrentGesture();
    const best = predictions[0] || null;
    const targetPrediction = predictions.find(item => item.label === target.id) || null;
    const targetScore = targetPrediction?.score || 0;

    let holdProgress = 0;
    let stableSuccess = false;
    let statusText = "Hold the target sign steady so the browser model sees a full 40-frame sequence.";

    if (best && best.label === target.id && best.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        stableSuccess = elapsed >= HOLD_SECONDS;
        statusText = stableSuccess
            ? `Matched ${target.title}. Moving to the next trained sign.`
            : `Matched ${target.title}. Keep holding it a little longer.`;
    } else if (best) {
        holdStartedAt = 0;
        statusText = `Top guess is ${prettifyLabel(best.label)} at ${Math.round(best.score * 100)}%. Adjust toward ${target.title}.`;
    } else {
        holdStartedAt = 0;
    }

    renderStatus(targetScore, holdProgress, statusText);

    if (stableSuccess && !lastSuccess) {
        lastSuccess = true;
        window.setTimeout(() => setGesture(currentGestureIndex + 1), 450);
        return;
    }
    lastSuccess = stableSuccess;
}

async function predictSequence() {
    if (predictionInFlight || frameBuffer.length < MAX_SEQUENCE || !modelState) {
        return;
    }
    predictionInFlight = true;
    try {
        const result = await predictWithBrowserModel(modelState, frameBuffer);
        renderPredictions(result.predictions);
        evaluatePredictions(result.predictions);
    } catch (error) {
        console.error(error);
        renderStatus(0, 0, `Prediction failed: ${error.message}`);
    } finally {
        predictionInFlight = false;
        await collectDiagnostics();
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

function onHolisticResults(results) {
    drawHolisticResults(canvasCtx, outputCanvas, inputVideo, results);
    frameBuffer.push(featureVectorFromResults(results));
    if (frameBuffer.length > MAX_SEQUENCE) {
        frameBuffer.shift();
    }
    trackedHands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % 6 === 0) {
        predictSequence().catch(console.error);
    }
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(0, 0, "Waiting for camera access.");
    await collectDiagnostics();

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
        holistic.onResults(onHolisticResults);
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
        lastCameraError = describeCameraError(error);
        cameraState.textContent = "Camera access failed";
        renderStatus(0, 0, lastCameraError);
    }
    await collectDiagnostics();
}

nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
calibrateBtn.addEventListener("click", () => loadModel().catch(console.error));

window.addEventListener("beforeunload", () => {
    stopLoop();
});

renderGestureCard();
renderPredictions([]);
renderStatus(0, 0, "Show the target sign to the camera.");
loadModel().then(() => startCamera()).catch(console.error);
