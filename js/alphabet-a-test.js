import {
    describeCameraError,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const DATASET_URL = "./datasets/landmarks_dataset.json?v=20260319-2";
const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.58;
const CANVAS_POINT_ORDER = [
    { id: "wrist", landmarkIndex: 0 },
    { id: "thumb", landmarkIndex: 4 },
    { id: "index", landmarkIndex: 8 },
    { id: "middle", landmarkIndex: 12 },
    { id: "pinky", landmarkIndex: 20 }
];
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

const labelSwitch = document.getElementById("dataset-label-switch");
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
let holistic = null;
let holdStartedAt = 0;
let cameraReady = false;
let trackedHands = 0;
let lastCameraError = "";
let permissionState = "prompt";
let datasetState = null;
let selectedLabel = "";
let latestMatch = null;

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function distance2D(left, right) {
    return Math.hypot(left.x - right.x, left.y - right.y);
}

function averagePoint(points) {
    return {
        x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
        y: points.reduce((sum, point) => sum + point.y, 0) / points.length
    };
}

function visiblePoint(point, minVisibility = 0.2) {
    return !!point && (point.visibility === undefined || point.visibility >= minVisibility);
}

function flattenPoints(points) {
    return points.flatMap(point => [point.x, point.y]);
}

function vectorDistance(left, right) {
    let total = 0;
    for (let index = 0; index < left.length; index += 2) {
        total += Math.hypot(left[index] - right[index], left[index + 1] - right[index + 1]);
    }
    return total / (left.length / 2);
}

function estimateHandScale(handLandmarks) {
    return Math.max(
        0.02,
        distance2D(handLandmarks[0], handLandmarks[9]),
        distance2D(handLandmarks[5], handLandmarks[17]),
        distance2D(handLandmarks[0], handLandmarks[17])
    );
}

function sortLandmarks(landmarks) {
    const sorted = Array.from(landmarks || []).sort((left, right) => left.id - right.id);
    return sorted.length >= 21 ? sorted.slice(0, 21) : null;
}

function normalizeHandToLocal(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return null;
    }

    const wrist = landmarks[0];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const pinkyMcp = landmarks[17];

    const sideRaw = {
        x: pinkyMcp.x - indexMcp.x,
        y: pinkyMcp.y - indexMcp.y
    };
    const sideLength = Math.hypot(sideRaw.x, sideRaw.y) || 1;
    const sideAxis = {
        x: sideRaw.x / sideLength,
        y: sideRaw.y / sideLength
    };
    let upAxis = {
        x: -sideAxis.y,
        y: sideAxis.x
    };
    const middleVector = {
        x: middleMcp.x - wrist.x,
        y: middleMcp.y - wrist.y
    };
    if (((middleVector.x * upAxis.x) + (middleVector.y * upAxis.y)) < 0) {
        upAxis = { x: -upAxis.x, y: -upAxis.y };
    }

    const scale = Math.max(
        0.02,
        (distance2D(indexMcp, pinkyMcp) + distance2D(wrist, middleMcp)) / 2
    );

    return landmarks.map(point => {
        const vector = {
            x: point.x - wrist.x,
            y: point.y - wrist.y
        };
        return {
            x: ((vector.x * sideAxis.x) + (vector.y * sideAxis.y)) / scale,
            y: ((vector.x * upAxis.x) + (vector.y * upAxis.y)) / scale
        };
    });
}

function getPrimaryHand(results) {
    const hands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean);
    if (!hands.length) {
        return { handsVisible: 0, handLandmarks: null };
    }

    const ranked = hands
        .map(handLandmarks => ({ handLandmarks, scale: estimateHandScale(handLandmarks) }))
        .sort((left, right) => right.scale - left.scale);

    return {
        handsVisible: hands.length,
        handLandmarks: ranked[0].handLandmarks
    };
}

function buildLiveVector(results) {
    const { handsVisible, handLandmarks } = getPrimaryHand(results);
    if (!handLandmarks) {
        return { handsVisible, localVector: null, rawHand: null };
    }

    const localPoints = normalizeHandToLocal(handLandmarks);
    return {
        handsVisible,
        localVector: flattenPoints(localPoints),
        rawHand: handLandmarks
    };
}

function computeLabelTolerance(templateVectors) {
    if (templateVectors.length <= 1) {
        return 0.26;
    }

    const pairDistances = [];
    for (let left = 0; left < templateVectors.length; left += 1) {
        for (let right = left + 1; right < templateVectors.length; right += 1) {
            pairDistances.push(vectorDistance(templateVectors[left], templateVectors[right]));
        }
    }

    const maxPair = Math.max(...pairDistances);
    return Math.max(0.24, Math.min(0.6, (maxPair * 0.9) + 0.08));
}

function buildDatasetState(dataset) {
    const byLabel = new Map();

    for (const sample of dataset.samples || []) {
        const hand = sortLandmarks(sample.hands?.[0]?.image_landmarks);
        if (!hand) {
            continue;
        }
        const localPoints = normalizeHandToLocal(hand);
        const vector = flattenPoints(localPoints);
        const label = sample.label;
        if (!byLabel.has(label)) {
            byLabel.set(label, []);
        }
        byLabel.get(label).push({
            id: sample.id,
            capturedAt: sample.captured_at,
            vector
        });
    }

    const labels = [...byLabel.keys()].sort((left, right) => left.localeCompare(right));
    const labelMap = new Map(labels.map(label => {
        const templates = byLabel.get(label);
        return [label, {
            label,
            templates,
            count: templates.length,
            tolerance: computeLabelTolerance(templates.map(template => template.vector))
        }];
    }));

    return { labels, byLabel: labelMap };
}

function currentLabelInfo() {
    return selectedLabel && datasetState ? datasetState.byLabel.get(selectedLabel) : null;
}

function renderLabelButtons() {
    labelSwitch.innerHTML = "";
    if (!datasetState?.labels?.length) {
        return;
    }

    datasetState.labels.forEach(label => {
        const info = datasetState.byLabel.get(label);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "gesture-mode-btn";
        button.textContent = `${label} (${info.count})`;
        if (label === selectedLabel) {
            button.classList.add("is-active");
        }
        button.addEventListener("click", () => setSelectedLabel(label));
        labelSwitch.appendChild(button);
    });
}

function renderStaticCopy() {
    const info = currentLabelInfo();
    letterTitle.textContent = selectedLabel || "Waiting";
    targetCopy.textContent = LABEL_DESCRIPTIONS[selectedLabel] || `Match the live hand to the JSON templates for ${selectedLabel}.`;
    datasetCountBadge.textContent = info ? `${info.count} samples` : "0 samples";
}

function renderBreakdown(match) {
    scoreBreakdown.innerHTML = "";
    const rows = match
        ? [
            { label: "JSON match", value: match.score },
            { label: "Best template distance", value: Math.max(0, 1 - (match.bestDistance / Math.max(match.tolerance, 0.001))) },
            { label: "Label tolerance", value: Math.max(0, 1 - match.tolerance) }
        ]
        : [
            { label: "JSON match", value: null },
            { label: "Best template distance", value: null },
            { label: "Label tolerance", value: null }
        ];

    rows.forEach(rowData => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const hasValue = typeof rowData.value === "number";
        const percent = hasValue ? Math.round(rowData.value * 100) : null;
        const tone = !hasValue ? "is-neutral" : percent >= 70 ? "is-good" : percent >= 40 ? "is-neutral" : "is-bad";
        const text = hasValue ? `${percent}%` : "Waiting";
        row.innerHTML = `<span>${rowData.label}</span><span class="gesture-check-badge ${tone}">${text}</span>`;
        scoreBreakdown.appendChild(row);
    });
}

function renderStatus(score, holdProgress, text) {
    const percent = Math.round(score * 100);
    gestureScore.textContent = `${percent}%`;
    gestureStatus.textContent = text;
    holdProgressBar.style.width = `${Math.round(holdProgress * 100)}%`;
    holdProgressValue.textContent = `${Math.round(holdProgress * 100)}%`;
    gestureScore.classList.toggle("is-strong", percent >= 70);
}

function renderDiagnostics() {
    diagnosticsList.innerHTML = "";
    const info = currentLabelInfo();
    const rows = [
        {
            label: "Matcher",
            value: "Only JSON templates are used. Live landmarks are normalized and compared directly to stored landmark samples.",
            badge: "JSON only",
            tone: "is-good"
        },
        {
            label: "Dataset",
            value: datasetState
                ? `Loaded ${datasetState.labels.length} labels from landmarks_dataset.json.`
                : "Dataset is not loaded yet.",
            badge: datasetState ? "Loaded" : "Waiting",
            tone: datasetState ? "is-good" : "is-neutral"
        },
        {
            label: "Selected label",
            value: info
                ? `${selectedLabel} with ${info.count} template sample(s), tolerance ${info.tolerance.toFixed(2)}.`
                : "No dataset label selected yet.",
            badge: selectedLabel || "None",
            tone: info ? "is-good" : "is-neutral"
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

    if (latestMatch) {
        rows.push({
            label: "Best match",
            value: `Distance ${latestMatch.bestDistance.toFixed(3)} against ${selectedLabel}.`,
            badge: `${Math.round(latestMatch.score * 100)}%`,
            tone: latestMatch.score >= 0.7 ? "is-good" : latestMatch.score >= 0.4 ? "is-neutral" : "is-bad"
        });
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

    diagnosticsSummary.textContent = info
        ? `Live hand is compared only to ${info.count} stored JSON template sample(s) for ${selectedLabel}.`
        : "Waiting for dataset label selection.";
}

function setSelectedLabel(label) {
    selectedLabel = label;
    holdStartedAt = 0;
    latestMatch = null;
    renderLabelButtons();
    renderStaticCopy();
    renderBreakdown(null);
    renderStatus(0, 0, `Show ${selectedLabel} to the camera.`);
    renderDiagnostics();
}

function scoreDatasetTemplate(localVector, labelInfo) {
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const template of labelInfo.templates) {
        bestDistance = Math.min(bestDistance, vectorDistance(localVector, template.vector));
    }
    return {
        bestDistance,
        tolerance: labelInfo.tolerance,
        score: clamp01(1 - (bestDistance / labelInfo.tolerance))
    };
}

function drawTracking(results) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    const { rawHand } = buildLiveVector(results);
    if (rawHand) {
        CANVAS_POINT_ORDER.forEach(({ id, landmarkIndex }) => {
            const point = rawHand[landmarkIndex];
            if (!point) {
                return;
            }
            canvasCtx.beginPath();
            canvasCtx.fillStyle = id === "wrist" ? "rgba(251, 191, 36, 0.98)" : "rgba(56, 189, 248, 0.98)";
            canvasCtx.arc(point.x * outputCanvas.width, point.y * outputCanvas.height, id === "wrist" ? 8 : 7, 0, Math.PI * 2);
            canvasCtx.fill();
        });
    }

    canvasCtx.restore();
}

function handleLiveFrame(results) {
    const { handsVisible, localVector } = buildLiveVector(results);
    trackedHands = handsVisible;

    if (!localVector) {
        latestMatch = null;
        holdStartedAt = 0;
        renderBreakdown(null);
        renderStatus(0, 0, selectedLabel ? `Show ${selectedLabel} with one clear hand.` : "Waiting for dataset.");
        renderDiagnostics();
        return;
    }

    const labelInfo = currentLabelInfo();
    if (!labelInfo) {
        latestMatch = null;
        holdStartedAt = 0;
        renderBreakdown(null);
        renderStatus(0, 0, "Waiting for dataset labels.");
        renderDiagnostics();
        return;
    }

    latestMatch = scoreDatasetTemplate(localVector, labelInfo);
    renderBreakdown(latestMatch);

    let holdProgress = 0;
    let statusText = `Adjust ${selectedLabel} until it matches the JSON templates.`;
    if (latestMatch.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        statusText = elapsed >= HOLD_SECONDS
            ? `${selectedLabel} matched.`
            : `Good ${selectedLabel}. Keep holding.`;
    } else {
        holdStartedAt = 0;
    }

    renderStatus(latestMatch.score, holdProgress, statusText);
    renderDiagnostics();
}

async function loadDataset() {
    const response = await fetch(DATASET_URL);
    if (!response.ok) {
        throw new Error(`Could not load landmarks_dataset.json (${response.status}).`);
    }
    const dataset = await response.json();
    datasetState = buildDatasetState(dataset);
    if (!datasetState.labels.length) {
        throw new Error("The dataset did not contain any hand landmark samples.");
    }
    setSelectedLabel(datasetState.labels.includes("A") ? "A" : datasetState.labels[0]);
}

async function refreshPermissionState() {
    permissionState = await getCameraPermissionState();
    renderDiagnostics();
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
    drawTracking(results);
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    handleLiveFrame(results);
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(0, 0, "Waiting for camera access.");
    await refreshPermissionState();

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
        lastCameraError = describeCameraError(error);
        cameraState.textContent = "Camera access failed";
        renderStatus(0, 0, lastCameraError);
        renderDiagnostics();
    }
}

retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => refreshPermissionState().catch(console.error));
window.addEventListener("beforeunload", () => stopLoop());

renderBreakdown(null);
renderDiagnostics();

Promise.all([loadDataset(), refreshPermissionState()])
    .then(() => startCamera())
    .catch(error => {
        lastCameraError = error.message;
        renderStatus(0, 0, error.message);
        renderDiagnostics();
    });
