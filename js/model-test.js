import {
    describeCameraError,
    getCameraPermissionState,
    prettifyLabel,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js";

const DATASET_URL = "./datasets/landmarks_dataset.json?v=20260319-5";
const K_NEIGHBORS = 5;
const Z_WEIGHT = 0.35;
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];
const DRAW_POINT_INDICES = [0, 4, 8, 12, 20];

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
let handsModel = null;
let datasetState = null;
let latestPrediction = null;
let cameraReady = false;

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
    const scale = Math.max(1e-6, ...centered.map(point => Math.hypot(point.x, point.y)));

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
    const samples = [];
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

        samples.push({
            id: sample.id,
            label: sample.label || "unknown",
            handedness: primaryHand.handedness || "Unknown",
            handednessScore: primaryHand.score || 0,
            vector: flattenLandmarks(normalized)
        });
        labelCounts.set(sample.label, (labelCounts.get(sample.label) || 0) + 1);
    }

    return {
        samples,
        sampleCount: samples.length,
        labelCount: labelCounts.size,
        labels: [...labelCounts.keys()].sort((left, right) => left.localeCompare(right))
    };
}

function formatTime(date) {
    return new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date);
}

function renderDiagnosticRows(rows) {
    diagnosticsList.innerHTML = "";
    rows.forEach(rowData => {
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
    });
}

async function updateDiagnostics(extra = {}) {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = extra.summary || "Everything is running directly in the browser with a JSON landmark matcher.";
    renderDiagnosticRows([
        {
            label: "Matcher",
            value: datasetState
                ? `Loaded ${datasetState.sampleCount} samples across ${datasetState.labelCount} labels from landmarks_dataset.json.`
                : "Dataset has not loaded yet.",
            badge: datasetState ? "Ready" : "Loading",
            tone: datasetState ? "is-good" : "is-neutral"
        },
        {
            label: "Camera state",
            value: cameraReady ? "Video frames are available for hand tracking." : "Camera stream is not ready yet.",
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
            value: latestPrediction ? "The matcher is receiving live hand landmarks." : "Waiting for one clear hand in frame.",
            badge: latestPrediction ? "Live" : "Idle",
            tone: latestPrediction ? "is-good" : "is-neutral"
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
        value.textContent = `vote ${item.vote.toFixed(3)} - d=${item.bestDistance.toFixed(3)}`;
        value.className = `gesture-check-badge ${index === 0 ? "is-good" : "is-neutral"}`;
        row.append(label, value);
        predictionList.appendChild(row);
    });
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
        return { handsVisible, primaryHand: null, prediction: null };
    }

    const normalized = normalizeLandmarks(primary.landmarks);
    if (!normalized) {
        return { handsVisible, primaryHand: primary.landmarks, prediction: null };
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
        return { handsVisible, primaryHand: primary.landmarks, prediction: null };
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

    return {
        handsVisible,
        primaryHand: primary.landmarks,
        prediction: {
            predictedLabel,
            confidence: totalWeight ? predictedWeight / totalWeight : 0,
            similarity: 1 / (1 + bestDistance),
            bestDistance,
            topMatches: rankedLabels.slice(0, 3).map(([label, vote]) => ({
                label,
                vote,
                bestDistance: labelBestDistances.get(label)
            })),
            liveHandedness: primary.handedness,
            liveHandednessScore: primary.handednessScore
        }
    };
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

function processResults(results) {
    const { handsVisible, primaryHand, prediction } = datasetState
        ? classifyCurrentHand(results)
        : { handsVisible: 0, primaryHand: null, prediction: null };

    latestPrediction = prediction;
    drawHand(primaryHand);
    bufferedFrames.textContent = `${handsVisible}`;
    cameraReady = true;
    cameraState.textContent = "Camera is live";

    if (prediction) {
        topPredictionLabel.textContent = prettifyLabel(prediction.predictedLabel);
        topPredictionScore.textContent = `Confidence ${Math.round(prediction.confidence * 100)}% | Similarity ${Math.round(prediction.similarity * 100)}%`;
        renderPredictions(prediction.topMatches);
        lastInferenceAt.textContent = formatTime(new Date());
    } else {
        topPredictionLabel.textContent = "Waiting for hand";
        topPredictionScore.textContent = "Show one hand clearly to start matching.";
        predictionList.innerHTML = "";
    }

    updateDiagnostics().catch(console.error);
}

async function checkModel() {
    modelStatusTitle.textContent = "Loading landmark dataset...";
    modelHealthBadge.textContent = "Loading";
    try {
        const response = await fetch(DATASET_URL);
        if (!response.ok) {
            throw new Error(`Could not load landmarks_dataset.json (${response.status}).`);
        }
        datasetState = buildDatasetState(await response.json());
        modelStatusTitle.textContent = "JSON matcher is ready";
        modelHealthBadge.textContent = `${datasetState.labelCount} labels`;
        topPredictionScore.textContent = "Show one hand clearly to start matching.";
        await updateDiagnostics();
    } catch (error) {
        console.error(error);
        datasetState = null;
        modelStatusTitle.textContent = "Dataset unavailable";
        modelHealthBadge.textContent = "Offline";
        await updateDiagnostics({ summary: `Could not load the landmark dataset: ${error.message}` });
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

async function startCamera() {
    cameraState.textContent = "Starting camera...";
    stopLoop();
    await updateDiagnostics();

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
        handsModel.onResults(processResults);
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
    latestPrediction = null;
    bufferedFrames.textContent = "0";
    topPredictionLabel.textContent = "Waiting for hand";
    topPredictionScore.textContent = "Show one hand clearly to start matching.";
    predictionList.innerHTML = "";
    lastInferenceAt.textContent = "Not run yet";
    await updateDiagnostics({
        summary: "The live matcher view was reset. Show one clear hand to produce the next match."
    });
});

window.addEventListener("beforeunload", () => {
    stopLoop();
});

checkModel()
    .then(() => startCamera())
    .catch(console.error);
