import {
    describeCameraError,
    drawHolisticResults,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.72;
const LETTER_A_POINTS = [0, 4, 8, 12, 20];
const FIST_TEMPLATE = [[0, 0], [-0.8, -0.15], [-0.35, -0.42], [0, -0.38], [0.55, -0.25]];

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
let latestDebug = null;

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function distance2D(left, right) {
    return Math.hypot(left.x - right.x, left.y - right.y);
}

function average(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
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

function normalizePoints(points) {
    const wrist = points[0];
    const centered = points.map(point => [point.x - wrist.x, point.y - wrist.y]);
    const distances = centered.slice(1).map(([x, y]) => Math.hypot(x, y));
    const scale = Math.max(0.0001, ...distances);
    return centered.map(([x, y]) => [x / scale, y / scale]);
}

function pairwiseVector(points) {
    const normalized = normalizePoints(points);
    const values = [];
    for (let i = 0; i < normalized.length; i += 1) {
        for (let j = i + 1; j < normalized.length; j += 1) {
            const dx = normalized[i][0] - normalized[j][0];
            const dy = normalized[i][1] - normalized[j][1];
            values.push(Math.hypot(dx, dy));
        }
    }
    return values;
}

const FIST_VECTOR = pairwiseVector(FIST_TEMPLATE.map(([x, y]) => ({ x, y })));

function extractSparseHand(handLandmarks) {
    if (!handLandmarks?.length) {
        return null;
    }
    return LETTER_A_POINTS.map(index => handLandmarks[index] || handLandmarks[0]);
}

function compareVectors(left, right, tolerance) {
    const diff = left.reduce((sum, value, index) => sum + Math.abs(value - right[index]), 0) / left.length;
    return Math.max(0, 1 - diff / tolerance);
}

function scoreProximity(actual, target, tolerance) {
    return clamp01(1 - Math.abs(actual - target) / tolerance);
}

function getBodyFrame(results) {
    const pose = results.poseLandmarks || [];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const shouldersVisible = visiblePoint(leftShoulder) && visiblePoint(rightShoulder);
    const center = shouldersVisible ? averagePoint([leftShoulder, rightShoulder]) : { x: 0.5, y: 0.52 };
    const scale = shouldersVisible ? Math.max(0.08, distance2D(leftShoulder, rightShoulder)) : 0.18;
    return { center, scale };
}

function scoreLetterAForHand(results, handLandmarks) {
    const wrist = handLandmarks[0];
    const thumbTip = handLandmarks[4];
    const indexMcp = handLandmarks[5];
    const middleMcp = handLandmarks[9];
    const ringMcp = handLandmarks[13];
    const pinkyMcp = handLandmarks[17];
    const indexTip = handLandmarks[8];
    const middleTip = handLandmarks[12];
    const ringTip = handLandmarks[16];
    const pinkyTip = handLandmarks[20];

    const sparse = extractSparseHand(handLandmarks);
    const fistScore = sparse ? compareVectors(pairwiseVector(sparse), FIST_VECTOR, 0.16) : 0;

    const handScale = Math.max(
        0.02,
        average([
            distance2D(wrist, indexMcp),
            distance2D(wrist, middleMcp),
            distance2D(wrist, ringMcp),
            distance2D(wrist, pinkyMcp)
        ])
    );

    const curledRatios = [
        distance2D(indexTip, indexMcp) / handScale,
        distance2D(middleTip, middleMcp) / handScale,
        distance2D(ringTip, ringMcp) / handScale,
        distance2D(pinkyTip, pinkyMcp) / handScale
    ];
    const curledScore = average(curledRatios.map(ratio => scoreProximity(ratio, 0.42, 0.38)));

    const thumbHorizontal = Math.abs((thumbTip.x - indexMcp.x) / handScale);
    const thumbVertical = Math.abs((thumbTip.y - indexMcp.y) / handScale);
    const thumbScore = (scoreProximity(thumbHorizontal, 0.58, 0.42) * 0.7) + (scoreProximity(thumbVertical, 0.12, 0.28) * 0.3);

    const bodyFrame = getBodyFrame(results);
    const wristX = (wrist.x - bodyFrame.center.x) / bodyFrame.scale;
    const wristY = (wrist.y - bodyFrame.center.y) / bodyFrame.scale;
    const bodyPositionScore = (scoreProximity(Math.abs(wristX), 0.55, 0.8) * 0.45) + (scoreProximity(wristY, 0.15, 0.7) * 0.55);

    const score = (fistScore * 0.4) + (curledScore * 0.25) + (thumbScore * 0.2) + (bodyPositionScore * 0.15);
    return {
        score,
        debug: {
            fistScore,
            curledScore,
            thumbScore,
            bodyPositionScore,
            wristX,
            wristY,
            thumbHorizontal,
            thumbVertical,
            curledRatios
        }
    };
}

function scoreLetterA(results) {
    const hands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean);
    if (!hands.length) {
        return { score: 0, debug: null, handsVisible: 0 };
    }
    const candidates = hands.map(hand => scoreLetterAForHand(results, hand)).sort((left, right) => right.score - left.score);
    return {
        score: candidates[0].score,
        debug: candidates[0].debug,
        handsVisible: hands.length
    };
}

function renderBreakdown(debug) {
    scoreBreakdown.innerHTML = "";
    const items = !debug
        ? [{ label: "Waiting", value: "No score yet.", badge: "Idle" }]
        : [
            { label: "Fist shape", value: `${Math.round(debug.fistScore * 100)}%`, badge: "Shape" },
            { label: "Finger curl", value: `${Math.round(debug.curledScore * 100)}%`, badge: "Curl" },
            { label: "Thumb placement", value: `${Math.round(debug.thumbScore * 100)}%`, badge: "Thumb" },
            { label: "Body position", value: `${Math.round(debug.bodyPositionScore * 100)}%`, badge: "Body" }
        ];

    items.forEach(item => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        row.innerHTML = `<span>${item.label}</span><span class="gesture-check-badge is-neutral">${item.value} ${item.badge ? `· ${item.badge}` : ""}</span>`;
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

function renderDiagnostics(permissionState) {
    diagnosticsList.innerHTML = "";
    const rows = [
        { label: "Matcher", value: "Geometric score for the letter A only.", badge: "A only", tone: "is-good" },
        { label: "Camera state", value: cameraReady ? "Live frames are being scored." : "Camera stream is not ready yet.", badge: cameraReady ? "Live" : "Waiting", tone: cameraReady ? "is-good" : "is-neutral" },
        { label: "Permission", value: `Camera permission state: ${permissionState}.`, badge: permissionState, tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral" },
        { label: "Tracked hands", value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible.", badge: `${trackedHands}`, tone: trackedHands ? "is-good" : "is-neutral" },
        { label: "Last camera error", value: lastCameraError || "No camera error recorded.", badge: lastCameraError ? "Has error" : "Clear", tone: lastCameraError ? "is-bad" : "is-good" }
    ];

    if (latestDebug) {
        rows.push({
            label: "Geometry debug",
            value: `Fist ${Math.round(latestDebug.fistScore * 100)}%, curl ${Math.round(latestDebug.curledScore * 100)}%, thumb ${Math.round(latestDebug.thumbScore * 100)}%, body ${Math.round(latestDebug.bodyPositionScore * 100)}%.`,
            badge: "Debug",
            tone: "is-good"
        });
    }

    rows.forEach(item => {
        const row = document.createElement("div");
        row.className = "gesture-diagnostic-row";
        row.innerHTML = `<div class="gesture-diagnostic-copy"><div class="gesture-diagnostic-label">${item.label}</div><div class="gesture-diagnostic-value">${item.value}</div></div><span class="gesture-check-badge ${item.tone || "is-neutral"}">${item.badge}</span>`;
        diagnosticsList.appendChild(row);
    });
}

async function collectDiagnostics(summary = "") {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = summary || "This page scores only the final visible pose for the ASL letter A.";
    renderDiagnostics(permissionState);
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

function handleScoredFrame(result) {
    latestDebug = result.debug;
    trackedHands = result.handsVisible;
    renderBreakdown(result.debug);

    let holdProgress = 0;
    let statusText = "Make the fist for A and hold it steady.";
    if (!result.handsVisible) {
        holdStartedAt = 0;
        statusText = "Show one clear hand to the camera.";
    } else if (result.handsVisible > 1) {
        holdStartedAt = 0;
        statusText = "Use one main hand for the A test.";
    } else if (result.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        statusText = elapsed >= HOLD_SECONDS
            ? "Letter A matched."
            : "Good A pose. Keep holding.";
    } else {
        holdStartedAt = 0;
        statusText = "Adjust your fist, thumb, and hand position.";
    }

    renderStatus(result.score, holdProgress, statusText);
    collectDiagnostics().catch(console.error);
}

function onHolisticResults(results) {
    drawHolisticResults(canvasCtx, outputCanvas, inputVideo, results);
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    handleScoredFrame(scoreLetterA(results));
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(0, 0, "Waiting for camera access.");
    renderBreakdown(null);
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
        lastCameraError = describeCameraError(error);
        cameraState.textContent = "Camera access failed";
        renderStatus(0, 0, lastCameraError);
        await collectDiagnostics("Camera failed to start.");
    }
}

retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
window.addEventListener("beforeunload", () => stopLoop());

renderBreakdown(null);
collectDiagnostics().then(() => startCamera()).catch(console.error);
