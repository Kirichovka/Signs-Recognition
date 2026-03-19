import {
    describeCameraError,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const DATASET_URL = "./datasets/landmarks_dataset.json?v=20260319-4";
const K_NEIGHBORS = 5;
const Z_WEIGHT = 0.35;
const SCORE_THRESHOLD = 0.55;
const HOLD_SECONDS = 1.0;
const DRAW_POINT_INDICES = [0, 4, 8, 12, 20];
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];
const LABEL_DESCRIPTIONS = {
    A: "Show the closed fist with the thumb resting along the side or front of the fist.",
    B: "Show the flat open hand with four straight fingers up and the thumb folded across the palm.",
    C: "Show the curved hand with a visible open C-shape between thumb and fingers.",
    D: "Show the index finger up, thumb touching the middle finger, other fingers folded.",
    E: "Show all fingertips curled tightly toward the thumb in a compact fist.",
    F: "Show the thumb and index touching to form a ring while the other fingers stay up.",
    G: "Show index and thumb pointing sideways in parallel, like a small horizontal pinch.",
    H: "Show index and middle extended together to the side while the other fingers fold.",
    I: "Show only the pinky extended while the other fingers stay curled.",
    J: "Show the J hand, starting like I and curving the pinky in a J motion."
};

const targetLetter = document.getElementById("target-letter");
const taskStep = document.getElementById("task-step");
const taskCopy = document.getElementById("task-copy");
const datasetBadge = document.getElementById("dataset-badge");
const prevTaskBtn = document.getElementById("prev-task-btn");
const nextTaskBtn = document.getElementById("next-task-btn");
const randomTaskBtn = document.getElementById("random-task-btn");
const confidenceScore = document.getElementById("confidence-score");
const similarityScore = document.getElementById("similarity-score");
const holdProgressBar = document.getElementById("hold-progress-bar");
const holdProgressValue = document.getElementById("hold-progress-value");
const classifierStatus = document.getElementById("classifier-status");
const statusBadge = document.getElementById("status-badge");
const predictionSummary = document.getElementById("prediction-summary");
const topMatches = document.getElementById("top-matches");
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
let cameraReady = false;
let trackedHands = 0;
let lastCameraError = "";
let permissionState = "prompt";
let datasetState = null;
let currentTaskIndex = 0;
let latestPrediction = null;
let holdStartedAt = 0;

function sortLandmarks(landmarks) {
    const sorted = Array.from(landmarks || []).sort((left, right) => left.id - right.id);
    return sorted.length >= 21 ? sorted.slice(0, 21) : null;
}

function normalizeLandmarks(landmarks) {
    if (!landmarks || landmarks.length < 21) {
        return null;
    }

    const wrist = landmarks[0];
    const centered = landmarks.map(point => ({
        x: point.x - wrist.x,
        y: point.y - wrist.y,
        z: point.z - wrist.z
    }));
    const scale = Math.max(
        1e-6,
        ...centered.map(point => Math.hypot(point.x, point.y))
    );

    return centered.map(point => ({
        x: point.x / scale,
        y: point.y / scale,
        z: (point.z / scale) * Z_WEIGHT
    }));
}

function flattenLandmarks(landmarks) {
    return landmarks.flatMap(point => [point.x, point.y, point.z]);
}

function mirrorVector(vector) {
    const mirrored = vector.slice();
    for (let index = 0; index < mirrored.length; index += 3) {
        mirrored[index] *= -1;
    }
    return mirrored;
}

function vectorDistance(leftVector, rightVector) {
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
    const normalizedSamples = [];
    const labelCounts = new Map();

    for (const sample of dataset.samples || []) {
        const sampleHands = (sample.hands || []).filter(hand => Array.isArray(hand.image_landmarks) && hand.image_landmarks.length >= 21);
        if (!sampleHands.length) {
            continue;
        }

        const primaryHand = sampleHands.reduce((best, hand) => {
            const score = hand.score || 0;
            return !best || score > (best.score || 0) ? hand : best;
        }, null);

        const normalized = normalizeLandmarks(sortLandmarks(primaryHand.image_landmarks));
        if (!normalized) {
            continue;
        }

        normalizedSamples.push({
            id: sample.id,
            label: sample.label || "unknown",
            handedness: primaryHand.handedness || "Unknown",
            handednessScore: primaryHand.score || 0,
            vector: flattenLandmarks(normalized)
        });
        labelCounts.set(sample.label, (labelCounts.get(sample.label) || 0) + 1);
    }

    const labels = [...labelCounts.keys()].sort((left, right) => left.localeCompare(right));
    return {
        sampleCount: normalizedSamples.length,
        labelCount: labelCounts.size,
        labelCounts,
        labels,
        samples: normalizedSamples
    };
}

function currentTargetLabel() {
    return datasetState?.labels?.[currentTaskIndex] || "";
}

function renderTask() {
    const target = currentTargetLabel();
    if (!target) {
        targetLetter.textContent = "Waiting";
        taskStep.textContent = "0 / 0";
        taskCopy.textContent = "Waiting for dataset labels.";
        return;
    }

    targetLetter.textContent = `Show ${target}`;
    taskStep.textContent = `${currentTaskIndex + 1} / ${datasetState.labels.length}`;
    taskCopy.textContent = LABEL_DESCRIPTIONS[target] || `Show the sign for ${target}.`;
}

function moveTask(delta) {
    if (!datasetState?.labels?.length) {
        return;
    }
    currentTaskIndex = (currentTaskIndex + delta + datasetState.labels.length) % datasetState.labels.length;
    holdStartedAt = 0;
    renderTask();
    renderPrediction(latestPrediction);
    renderDiagnostics(latestPrediction);
}

function randomTask() {
    if (!datasetState?.labels?.length) {
        return;
    }
    if (datasetState.labels.length === 1) {
        currentTaskIndex = 0;
    } else {
        let nextIndex = currentTaskIndex;
        while (nextIndex === currentTaskIndex) {
            nextIndex = Math.floor(Math.random() * datasetState.labels.length);
        }
        currentTaskIndex = nextIndex;
    }
    holdStartedAt = 0;
    renderTask();
    renderPrediction(latestPrediction);
    renderDiagnostics(latestPrediction);
}

function getPrimaryLiveHand(results) {
    const liveHands = (results.multiHandLandmarks || []).map((landmarks, index) => {
        const classification = results.multiHandedness?.[index]?.classification?.[0];
        return {
            landmarks,
            handedness: classification?.label || "Unknown",
            handednessScore: classification?.score || 0
        };
    });

    if (!liveHands.length) {
        return { handsVisible: 0, primary: null };
    }

    const primary = liveHands.reduce((best, hand) => {
        return !best || hand.handednessScore > best.handednessScore ? hand : best;
    }, null);

    return {
        handsVisible: liveHands.length,
        primary
    };
}

function classifyCurrentHand(results) {
    const { handsVisible, primary } = getPrimaryLiveHand(results);
    if (!primary) {
        return { handsVisible, prediction: null, primaryHand: null };
    }

    const normalized = normalizeLandmarks(primary.landmarks);
    if (!normalized) {
        return { handsVisible, prediction: null, primaryHand: primary.landmarks };
    }

    const queryVector = flattenLandmarks(normalized);
    const queryMirrored = mirrorVector(queryVector);

    const distances = datasetState.samples.map(sample => {
        const distance = Math.min(
            vectorDistance(queryVector, sample.vector),
            vectorDistance(queryMirrored, sample.vector)
        );
        return {
            label: sample.label,
            distance,
            sampleId: sample.id
        };
    }).sort((left, right) => left.distance - right.distance);

    if (!distances.length) {
        return { handsVisible, prediction: null, primaryHand: primary.landmarks };
    }

    const neighbors = distances.slice(0, Math.min(K_NEIGHBORS, distances.length));
    const labelScores = new Map();
    const labelBestDistances = new Map();

    neighbors.forEach(neighbor => {
        const vote = Math.exp(-4 * neighbor.distance);
        labelScores.set(neighbor.label, (labelScores.get(neighbor.label) || 0) + vote);
        const previousBest = labelBestDistances.get(neighbor.label);
        if (previousBest === undefined || neighbor.distance < previousBest) {
            labelBestDistances.set(neighbor.label, neighbor.distance);
        }
    });

    const rankedLabels = [...labelScores.entries()].sort((left, right) => right[1] - left[1]);
    const [predictedLabel, predictedWeight] = rankedLabels[0];
    const totalWeight = [...labelScores.values()].reduce((sum, value) => sum + value, 0);
    const bestDistance = labelBestDistances.get(predictedLabel);
    const confidence = totalWeight ? predictedWeight / totalWeight : 0;
    const similarity = 1 / (1 + bestDistance);

    const topMatches = rankedLabels.slice(0, 3).map(([label, vote]) => ({
        label,
        vote,
        bestDistance: labelBestDistances.get(label)
    }));

    return {
        handsVisible,
        primaryHand: primary.landmarks,
        prediction: {
            predictedLabel,
            confidence,
            similarity,
            bestDistance,
            topMatches,
            liveHandedness: primary.handedness,
            liveHandednessScore: primary.handednessScore
        }
    };
}

function renderPrediction(prediction) {
    const target = currentTargetLabel();
    if (!prediction || !target) {
        confidenceScore.textContent = "0%";
        similarityScore.textContent = "0%";
        classifierStatus.textContent = target ? `Show ${target} to start the check.` : "Waiting for a hand in the frame.";
        predictionSummary.textContent = "Predicted label: waiting.";
        statusBadge.textContent = "Idle";
        statusBadge.className = "gesture-check-badge is-neutral";
        holdProgressBar.style.width = "0%";
        holdProgressValue.textContent = "0%";
        topMatches.innerHTML = "";

        [
            ["Top match 1", "Waiting"],
            ["Top match 2", "Waiting"],
            ["Top match 3", "Waiting"]
        ].forEach(([label, value]) => {
            const row = document.createElement("div");
            row.className = "gesture-check-row";
            row.innerHTML = `<span>${label}</span><span class="gesture-check-badge is-neutral">${value}</span>`;
            topMatches.appendChild(row);
        });
        return;
    }

    const confidencePercent = Math.round(prediction.confidence * 100);
    const similarityPercent = Math.round(prediction.similarity * 100);
    const matchesTarget = prediction.predictedLabel === target;
    confidenceScore.textContent = `${matchesTarget ? confidencePercent : 0}%`;
    similarityScore.textContent = `${matchesTarget ? similarityPercent : 0}%`;
    predictionSummary.textContent = `Predicted label: ${prediction.predictedLabel}.`;

    let holdProgress = 0;
    if (matchesTarget && prediction.confidence >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        classifierStatus.textContent = elapsed >= HOLD_SECONDS
            ? `Correct: ${target}.`
            : `Good ${target}. Keep holding.`;
    } else {
        holdStartedAt = 0;
        classifierStatus.textContent = matchesTarget
            ? `This looks like ${target}, but confidence is still low.`
            : `Current guess is ${prediction.predictedLabel}. Show ${target}.`;
    }

    holdProgressBar.style.width = `${Math.round(holdProgress * 100)}%`;
    holdProgressValue.textContent = `${Math.round(holdProgress * 100)}%`;

    if (matchesTarget && prediction.confidence >= 0.7) {
        statusBadge.textContent = "Strong";
        statusBadge.className = "gesture-check-badge is-good";
    } else if (matchesTarget) {
        statusBadge.textContent = "Maybe";
        statusBadge.className = "gesture-check-badge is-neutral";
    } else {
        statusBadge.textContent = "Wrong";
        statusBadge.className = "gesture-check-badge is-bad";
    }

    topMatches.innerHTML = "";
    prediction.topMatches.forEach((match, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const tone = match.label === target ? "is-good" : index === 0 ? "is-neutral" : "is-neutral";
        row.innerHTML = `<span>${index + 1}. ${match.label}</span><span class="gesture-check-badge ${tone}">vote ${match.vote.toFixed(3)} - d=${match.bestDistance.toFixed(3)}</span>`;
        topMatches.appendChild(row);
    });
}

function renderDiagnostics(prediction) {
    diagnosticsList.innerHTML = "";
    const target = currentTargetLabel();

    const rows = [
        {
            label: "Task",
            value: target ? `Current task: show ${target}.` : "No target label is active yet.",
            badge: target || "Waiting",
            tone: target ? "is-good" : "is-neutral"
        },
        {
            label: "Matcher",
            value: "Primary hand by handedness score, wrist normalization, mirrored x vector, kNN over all samples, weighted vote by exp(-4 * distance).",
            badge: "JSON kNN",
            tone: "is-good"
        },
        {
            label: "Dataset",
            value: datasetState
                ? `Loaded ${datasetState.sampleCount} samples across ${datasetState.labelCount} labels.`
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
                value: `${prediction.liveHandedness} with handedness score ${Math.round(prediction.liveHandednessScore * 100)}%.`,
                badge: prediction.liveHandedness,
                tone: "is-good"
            },
            {
                label: "Best guess",
                value: `${prediction.predictedLabel}, best distance ${prediction.bestDistance.toFixed(3)}, similarity ${Math.round(prediction.similarity * 100)}%.`,
                badge: prediction.predictedLabel,
                tone: prediction.predictedLabel === target ? "is-good" : "is-bad"
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
        ? `Task ${target}. Best guess is ${prediction.predictedLabel}; confidence and similarity come from the 5 nearest JSON samples.`
        : target
            ? `Task ${target}. Waiting for a live hand to classify.`
            : "Waiting for dataset labels.";
}

function drawHand(rawHand) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(inputVideo, 0, 0, outputCanvas.width, outputCanvas.height);

    if (rawHand) {
        HAND_CONNECTIONS.forEach(([startIndex, endIndex]) => {
            const start = rawHand[startIndex];
            const end = rawHand[endIndex];
            if (!start || !end) {
                return;
            }
            canvasCtx.beginPath();
            canvasCtx.moveTo(start.x * outputCanvas.width, start.y * outputCanvas.height);
            canvasCtx.lineTo(end.x * outputCanvas.width, end.y * outputCanvas.height);
            canvasCtx.strokeStyle = "rgba(0, 200, 255, 0.78)";
            canvasCtx.lineWidth = 2;
            canvasCtx.stroke();
        });

        rawHand.forEach((landmark, index) => {
            canvasCtx.beginPath();
            canvasCtx.fillStyle = index === 0 ? "rgba(251, 191, 36, 0.98)" : DRAW_POINT_INDICES.includes(index) ? "rgba(56, 189, 248, 0.98)" : "rgba(30, 255, 30, 0.82)";
            canvasCtx.arc(landmark.x * outputCanvas.width, landmark.y * outputCanvas.height, index === 0 ? 6 : 4, 0, Math.PI * 2);
            canvasCtx.fill();
        });
    }

    canvasCtx.restore();
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
    datasetBadge.textContent = `${datasetState.sampleCount} samples`;
    currentTaskIndex = 0;
    renderTask();
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
    cameraState.textContent = "Camera is live";

    const { handsVisible, primaryHand, prediction } = datasetState
        ? classifyCurrentHand(results)
        : { handsVisible: 0, primaryHand: null, prediction: null };

    trackedHands = handsVisible;
    latestPrediction = prediction;
    drawHand(primaryHand);
    renderPrediction(prediction);
    renderDiagnostics(prediction);
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderPrediction(null);
    await refreshPermissionState();

    if (!handsModel) {
        handsModel = new globalThis.Hands({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        handsModel.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
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
        renderPrediction(null);
        classifierStatus.textContent = lastCameraError;
        statusBadge.textContent = "Error";
        statusBadge.className = "gesture-check-badge is-bad";
        renderDiagnostics(null);
    }
}

prevTaskBtn.addEventListener("click", () => moveTask(-1));
nextTaskBtn.addEventListener("click", () => moveTask(1));
randomTaskBtn.addEventListener("click", randomTask);
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => refreshPermissionState().catch(console.error));
window.addEventListener("beforeunload", () => stopLoop());

renderTask();
renderPrediction(null);
renderDiagnostics(null);

Promise.all([loadDataset(), refreshPermissionState()])
    .then(() => startCamera())
    .catch(error => {
        lastCameraError = error.message;
        renderPrediction(null);
        classifierStatus.textContent = error.message;
        statusBadge.textContent = "Error";
        statusBadge.className = "gesture-check-badge is-bad";
        renderDiagnostics(null);
    });
