import {
    describeCameraError,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const DATASET_URL = "./datasets/landmarks_dataset.json?v=20260319-3";
const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.55;
const K_NEIGHBORS = 5;
const Z_WEIGHT = 0.35;
const DRAW_POINT_INDICES = [0, 4, 8, 12, 20];
const LABEL_DESCRIPTIONS = {
    A: "Closed fist with the thumb resting along the side or front of the fist.",
    B: "Flat open hand with four straight fingers up and the thumb folded across the palm.",
    C: "Curved hand with a visible open C-shape between thumb and fingers.",
    D: "Index finger up, thumb touching the middle finger, other fingers folded.",
    E: "All fingertips curled tightly toward the thumb in a compact fist.",
    F: "Thumb and index touch to form a ring while the other fingers stay up.",
    G: "Index and thumb point sideways in parallel, like a small horizontal pinch.",
    H: "Index and middle extend together to the side while the other fingers fold.",
    I: "Only the pinky extends while the other fingers stay curled.",
    J: "Start from I and sweep the pinky in a J motion."
};

const letterTitle = document.getElementById("letter-title");
const targetCopy = document.getElementById("target-copy");
const datasetCountBadge = document.getElementById("dataset-count-badge");
const gestureScore = document.getElementById("gesture-score");
const gestureStatus = document.getElementById("gesture-status");
const holdProgressBar = document.getElementById("hold-progress-bar");
const holdProgressValue = document.getElementById("hold-progress-value");
const scoreBreakdown = document.getElementById("score-breakdown");
const cameraState = document.getElementById("camera-state");
const retryCameraBtn = document.getElementById("retry-camera-btn");
const refreshDiagnosticsBtn = document.getElementById("refresh-diagnostics-btn");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsList = document.getElementById("diagnostics-list");
const inputVideo = document.getElementById("input-video");
const outputCanvas = document.getElementById("output-canvas");

const canvasCtx = outputCanvas.getContext("2d");

let activeStream = null;
let animationFrameId = 0;
let handsModel = null;
let holdStartedAt = 0;
let holdLabel = "";
let cameraReady = false;
let trackedHands = 0;
let lastCameraError = "";
let permissionState = "prompt";
let datasetState = null;
let latestPrediction = null;

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function sortLandmarks(landmarks) {
    const sorted = Array.from(landmarks || []).sort((left, right) => left.id - right.id);
    return sorted.length >= 21 ? sorted.slice(0, 21) : null;
}

function normalizeLandmarks(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return null;
    }

    const wrist = landmarks[0];
    const relative = landmarks.map(point => ({
        x: point.x - wrist.x,
        y: point.y - wrist.y,
        z: point.z - wrist.z
    }));
    const scale = Math.max(
        0.0001,
        ...relative.map(point => Math.hypot(point.x, point.y))
    );

    return relative.map(point => ({
        x: point.x / scale,
        y: point.y / scale,
        z: (point.z / scale) * Z_WEIGHT
    }));
}

function mirrorNormalizedLandmarks(landmarks) {
    return landmarks.map(point => ({
        x: -point.x,
        y: point.y,
        z: point.z
    }));
}

function flattenNormalizedLandmarks(landmarks) {
    return landmarks.flatMap(point => [point.x, point.y, point.z]);
}

function landmarkDistance(leftVector, rightVector) {
    let total = 0;
    const pointCount = leftVector.length / 3;
    for (let index = 0; index < leftVector.length; index += 3) {
        total += Math.hypot(
            leftVector[index] - rightVector[index],
            leftVector[index + 1] - rightVector[index + 1],
            leftVector[index + 2] - rightVector[index + 2]
        );
    }
    return total / pointCount;
}

function buildDatasetState(dataset) {
    const labelsWithSamples = new Set();
    const normalizedSamples = [];

    for (const sample of dataset.samples || []) {
        const hands = (sample.hands || [])
            .map(hand => ({
                handedness: hand.handedness || "Unknown",
                score: Number.isFinite(hand.score) ? hand.score : 0,
                landmarks: sortLandmarks(hand.image_landmarks)
            }))
            .filter(hand => hand.landmarks);

        if (!hands.length) {
            continue;
        }

        const primaryHand = hands.sort((left, right) => right.score - left.score)[0];
        const normalized = normalizeLandmarks(primaryHand.landmarks);
        if (!normalized) {
            continue;
        }

        labelsWithSamples.add(sample.label);
        normalizedSamples.push({
            id: sample.id,
            label: sample.label,
            capturedAt: sample.captured_at,
            handedness: primaryHand.handedness,
            handednessScore: primaryHand.score,
            vector: flattenNormalizedLandmarks(normalized)
        });
    }

    return {
        labels: [...labelsWithSamples].sort((left, right) => left.localeCompare(right)),
        sampleCount: normalizedSamples.length,
        samples: normalizedSamples
    };
}

function getPrimaryLiveHand(results) {
    const liveHands = (results.multiHandLandmarks || []).map((landmarks, index) => {
        const classification = results.multiHandedness?.[index]?.classification?.[0];
        return {
            landmarks,
            handedness: classification?.label || "Unknown",
            handednessScore: Number.isFinite(classification?.score) ? classification.score : 0
        };
    });

    if (!liveHands.length) {
        return { handsVisible: 0, primary: null };
    }

    const primary = [...liveHands].sort((left, right) => right.handednessScore - left.handednessScore)[0];
    return {
        handsVisible: liveHands.length,
        primary
    };
}

function classifyLiveHand(results) {
    const { handsVisible, primary } = getPrimaryLiveHand(results);
    if (!primary) {
        return { handsVisible, prediction: null, rawHand: null };
    }

    const normalized = normalizeLandmarks(primary.landmarks);
    const mirrored = mirrorNormalizedLandmarks(normalized);
    const normalVector = flattenNormalizedLandmarks(normalized);
    const mirroredVector = flattenNormalizedLandmarks(mirrored);

    const distances = datasetState.samples
        .map(sample => {
            const normalDistance = landmarkDistance(normalVector, sample.vector);
            const mirroredDistance = landmarkDistance(mirroredVector, sample.vector);
            return {
                id: sample.id,
                label: sample.label,
                handedness: sample.handedness,
                handednessScore: sample.handednessScore,
                distance: Math.min(normalDistance, mirroredDistance)
            };
        })
        .sort((left, right) => left.distance - right.distance);

    const neighbors = distances.slice(0, Math.min(K_NEIGHBORS, distances.length)).map(neighbor => ({
        ...neighbor,
        vote: Math.exp(-4 * neighbor.distance)
    }));

    const votesByLabel = new Map();
    for (const neighbor of neighbors) {
        votesByLabel.set(neighbor.label, (votesByLabel.get(neighbor.label) || 0) + neighbor.vote);
    }

    const winner = [...votesByLabel.entries()].sort((left, right) => right[1] - left[1])[0];
    const totalVotes = neighbors.reduce((sum, neighbor) => sum + neighbor.vote, 0);
    const bestDistance = neighbors.length ? neighbors[0].distance : Number.POSITIVE_INFINITY;
    const winnerLabel = winner?.[0] || "Unknown";
    const winnerVotes = winner?.[1] || 0;
    const confidence = totalVotes ? winnerVotes / totalVotes : 0;
    const similarity = Number.isFinite(bestDistance) ? 1 / (1 + bestDistance) : 0;

    return {
        handsVisible,
        rawHand: primary.landmarks,
        prediction: {
            label: winnerLabel,
            confidence,
            similarity,
            bestDistance,
            neighbors,
            liveHandedness: primary.handedness,
            liveHandednessScore: primary.handednessScore
        }
    };
}

function renderStaticCopy(prediction) {
    if (!prediction) {
        letterTitle.textContent = "Waiting";
        targetCopy.textContent = "Show one clear hand to the camera and the page will classify it against all saved JSON samples.";
        return;
    }

    letterTitle.textContent = prediction.label;
    targetCopy.textContent = LABEL_DESCRIPTIONS[prediction.label] || `Nearest-neighbor match for ${prediction.label}.`;
}

function renderBreakdown(prediction) {
    scoreBreakdown.innerHTML = "";

    const rows = !prediction
        ? [
            { label: "Best label", value: "Waiting", tone: "is-neutral" },
            { label: "Confidence", value: "Waiting", tone: "is-neutral" },
            { label: "Similarity", value: "Waiting", tone: "is-neutral" }
        ]
        : [
            {
                label: "Best label",
                value: prediction.label,
                tone: "is-good"
            },
            {
                label: "Confidence",
                value: `${Math.round(prediction.confidence * 100)}%`,
                tone: prediction.confidence >= 0.7 ? "is-good" : prediction.confidence >= 0.4 ? "is-neutral" : "is-bad"
            },
            {
                label: "Similarity",
                value: `${Math.round(prediction.similarity * 100)}%`,
                tone: prediction.similarity >= 0.7 ? "is-good" : prediction.similarity >= 0.4 ? "is-neutral" : "is-bad"
            },
            {
                label: "Nearest 1",
                value: prediction.neighbors[0] ? `${prediction.neighbors[0].label} · d=${prediction.neighbors[0].distance.toFixed(3)}` : "Missing",
                tone: prediction.neighbors[0] ? "is-good" : "is-neutral"
            },
            {
                label: "Nearest 2",
                value: prediction.neighbors[1] ? `${prediction.neighbors[1].label} · d=${prediction.neighbors[1].distance.toFixed(3)}` : "Missing",
                tone: prediction.neighbors[1] ? "is-neutral" : "is-neutral"
            },
            {
                label: "Nearest 3",
                value: prediction.neighbors[2] ? `${prediction.neighbors[2].label} · d=${prediction.neighbors[2].distance.toFixed(3)}` : "Missing",
                tone: prediction.neighbors[2] ? "is-neutral" : "is-neutral"
            }
        ];

    rows.forEach(rowData => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        row.innerHTML = `<span>${rowData.label}</span><span class="gesture-check-badge ${rowData.tone}">${rowData.value}</span>`;
        scoreBreakdown.appendChild(row);
    });
}

function renderStatus(prediction, holdProgress, text) {
    const confidencePercent = Math.round((prediction?.confidence || 0) * 100);
    gestureScore.textContent = `${confidencePercent}%`;
    gestureStatus.textContent = text;
    holdProgressBar.style.width = `${Math.round(holdProgress * 100)}%`;
    holdProgressValue.textContent = `${Math.round(holdProgress * 100)}%`;
    gestureScore.classList.toggle("is-strong", confidencePercent >= 70);
}

function renderDiagnostics(prediction) {
    diagnosticsList.innerHTML = "";

    const rows = [
        {
            label: "Matcher",
            value: "kNN over all JSON samples with mirrored hand matching and weighted label voting.",
            badge: "kNN",
            tone: "is-good"
        },
        {
            label: "Dataset",
            value: datasetState
                ? `Loaded ${datasetState.labels.length} labels and ${datasetState.sampleCount} samples from landmarks_dataset.json.`
                : "Dataset is not loaded yet.",
            badge: datasetState ? "Loaded" : "Waiting",
            tone: datasetState ? "is-good" : "is-neutral"
        },
        {
            label: "Camera state",
            value: cameraReady ? "Live frames are being processed." : "Camera stream is not ready yet.",
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
            value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible.",
            badge: `${trackedHands}`,
            tone: trackedHands ? "is-good" : "is-neutral"
        }
    ];

    if (prediction) {
        rows.push(
            {
                label: "Primary hand",
                value: `${prediction.liveHandedness} (${Math.round(prediction.liveHandednessScore * 100)}% handedness score).`,
                badge: prediction.liveHandedness,
                tone: "is-good"
            },
            {
                label: "Prediction",
                value: `${prediction.label} with confidence ${Math.round(prediction.confidence * 100)}% and similarity ${Math.round(prediction.similarity * 100)}%.`,
                badge: prediction.label,
                tone: prediction.confidence >= 0.7 ? "is-good" : prediction.confidence >= 0.4 ? "is-neutral" : "is-bad"
            }
        );
    }

    if (lastCameraError) {
        rows.push({
            label: "Last camera error",
            value: lastCameraError,
            badge: "Has error",
            tone: "is-bad"
        });
    }

    rows.forEach(item => {
        const row = document.createElement("div");
        row.className = "gesture-diagnostic-row";
        row.innerHTML = `<div class="gesture-diagnostic-copy"><div class="gesture-diagnostic-label">${item.label}</div><div class="gesture-diagnostic-value">${item.value}</div></div><span class="gesture-check-badge ${item.tone}">${item.badge}</span>`;
        diagnosticsList.appendChild(row);
    });

    diagnosticsSummary.textContent = prediction
        ? `Best label ${prediction.label}. Top 3 nearest matches are shown in the breakdown.`
        : "Waiting for a live hand to classify against the JSON dataset.";
}

function drawTracking(rawHand) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(inputVideo, 0, 0, outputCanvas.width, outputCanvas.height);

    if (rawHand) {
        DRAW_POINT_INDICES.forEach(index => {
            const point = rawHand[index];
            if (!point) {
                return;
            }
            canvasCtx.beginPath();
            canvasCtx.fillStyle = index === 0 ? "rgba(251, 191, 36, 0.98)" : "rgba(56, 189, 248, 0.98)";
            canvasCtx.arc(point.x * outputCanvas.width, point.y * outputCanvas.height, index === 0 ? 8 : 7, 0, Math.PI * 2);
            canvasCtx.fill();
        });
    }

    canvasCtx.restore();
}

function handlePredictionFrame(prediction) {
    let holdProgress = 0;
    let statusText = "Show one clear hand to classify.";

    if (!prediction) {
        holdStartedAt = 0;
        holdLabel = "";
        renderStatus(null, 0, statusText);
        renderStaticCopy(null);
        renderBreakdown(null);
        renderDiagnostics(null);
        return;
    }

    const stableLabel = holdLabel === prediction.label;
    if (prediction.confidence >= SCORE_THRESHOLD && stableLabel) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
    } else if (prediction.confidence >= SCORE_THRESHOLD) {
        holdLabel = prediction.label;
        holdStartedAt = performance.now();
    } else {
        holdLabel = prediction.label;
        holdStartedAt = 0;
    }

    if (holdStartedAt) {
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        statusText = elapsed >= HOLD_SECONDS
            ? `Recognized as ${prediction.label}.`
            : `Best label ${prediction.label}. Keep holding.`;
    } else {
        statusText = `Best label ${prediction.label}. Raise confidence by matching the nearest samples more closely.`;
    }

    renderStatus(prediction, holdProgress, statusText);
    renderStaticCopy(prediction);
    renderBreakdown(prediction);
    renderDiagnostics(prediction);
}

async function loadDataset() {
    const response = await fetch(DATASET_URL);
    if (!response.ok) {
        throw new Error(`Could not load landmarks_dataset.json (${response.status}).`);
    }
    const dataset = await response.json();
    datasetState = buildDatasetState(dataset);
    if (!datasetState.sampleCount) {
        throw new Error("The dataset did not contain any usable hand samples.");
    }
    datasetCountBadge.textContent = `${datasetState.sampleCount} total samples`;
}

async function refreshPermissionState() {
    permissionState = await getCameraPermissionState();
    renderDiagnostics(latestPrediction);
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

function onHandsResults(results) {
    cameraReady = true;
    trackedHands = results.multiHandLandmarks?.length || 0;
    cameraState.textContent = "Camera is live";

    const { prediction, rawHand } = datasetState ? classifyLiveHand(results) : { prediction: null, rawHand: null };
    latestPrediction = prediction;
    drawTracking(rawHand);
    handlePredictionFrame(prediction);
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(null, 0, "Waiting for camera access.");
    await refreshPermissionState();

    if (!handsModel) {
        handsModel = new globalThis.Hands({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        handsModel.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        handsModel.onResults(onHandsResults);
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
            await handsModel.send({ image: inputVideo });
            animationFrameId = requestAnimationFrame(processFrame);
        };
        animationFrameId = requestAnimationFrame(processFrame);
    } catch (error) {
        lastCameraError = describeCameraError(error);
        cameraState.textContent = "Camera access failed";
        renderStatus(null, 0, lastCameraError);
        renderDiagnostics(null);
    }
}

retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => refreshPermissionState().catch(console.error));
window.addEventListener("beforeunload", () => stopLoop());

renderBreakdown(null);
renderDiagnostics(null);

Promise.all([loadDataset(), refreshPermissionState()])
    .then(() => startCamera())
    .catch(error => {
        lastCameraError = error.message;
        renderStatus(null, 0, error.message);
        renderDiagnostics(null);
    });
