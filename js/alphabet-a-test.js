import {
    describeCameraError,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.55;
const CALIBRATION_STORAGE_KEY = "gesture-trainer.sync-letter-a.v2";
const POINT_TOLERANCES = {
    wrist: 0.48,
    thumb: 0.24,
    index: 0.2,
    middle: 0.2,
    pinky: 0.22
};

const SYNC_STEPS = [
    {
        id: "wrist",
        landmarkIndex: 0,
        title: "Show the base of the fist",
        description: "Hold the A pose and make sure the wrist point is steady before capturing."
    },
    {
        id: "thumb",
        landmarkIndex: 4,
        title: "Show the thumb",
        description: "Keep the thumb visible on the front or side of the fist, then capture it."
    },
    {
        id: "index",
        landmarkIndex: 8,
        title: "Show the index finger",
        description: "Keep the index fingertip tucked in the fist and capture its exact position."
    },
    {
        id: "middle",
        landmarkIndex: 12,
        title: "Show the middle finger",
        description: "Keep the middle fingertip in the closed fist and capture it."
    },
    {
        id: "pinky",
        landmarkIndex: 20,
        title: "Show the pinky",
        description: "Keep the pinky tucked in and capture the last point to finish synchronization."
    }
];

const CANVAS_POINT_ORDER = [
    { id: "wrist", landmarkIndex: 0 },
    { id: "thumb", landmarkIndex: 4 },
    { id: "index", landmarkIndex: 8 },
    { id: "middle", landmarkIndex: 12 },
    { id: "pinky", landmarkIndex: 20 }
];

const calibrationStatus = document.getElementById("calibration-status");
const syncStepBadge = document.getElementById("sync-step-badge");
const syncStepTitle = document.getElementById("sync-step-title");
const syncStepCopy = document.getElementById("sync-step-copy");
const startSyncBtn = document.getElementById("start-sync-btn");
const captureSyncBtn = document.getElementById("capture-sync-btn");
const resetSyncBtn = document.getElementById("reset-sync-btn");
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
let latestSample = null;
let latestPointScores = null;
let savedCalibration = loadCalibration();
let syncSession = null;

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

function scoreDistance(left, right, tolerance) {
    const distance = Math.hypot(left.x - right.x, left.y - right.y);
    return clamp01(1 - (distance / tolerance));
}

function clonePoint(point) {
    return { x: point.x, y: point.y };
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

function estimateHandScale(handLandmarks) {
    return Math.max(
        0.02,
        distance2D(handLandmarks[0], handLandmarks[9]),
        distance2D(handLandmarks[0], handLandmarks[17])
    );
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

function normalizeToBody(point, bodyFrame) {
    return {
        x: (point.x - bodyFrame.center.x) / bodyFrame.scale,
        y: (point.y - bodyFrame.center.y) / bodyFrame.scale
    };
}

function normalizeToWrist(point, wrist, bodyFrame) {
    return {
        x: (point.x - wrist.x) / bodyFrame.scale,
        y: (point.y - wrist.y) / bodyFrame.scale
    };
}

function buildLiveSample(results) {
    const { handsVisible, handLandmarks } = getPrimaryHand(results);
    if (!handLandmarks) {
        return { handsVisible, sample: null };
    }

    const bodyFrame = getBodyFrame(results);
    const wrist = handLandmarks[0];
    const sample = {
        wristBody: normalizeToBody(wrist, bodyFrame),
        thumb: normalizeToWrist(handLandmarks[4], wrist, bodyFrame),
        index: normalizeToWrist(handLandmarks[8], wrist, bodyFrame),
        middle: normalizeToWrist(handLandmarks[12], wrist, bodyFrame),
        pinky: normalizeToWrist(handLandmarks[20], wrist, bodyFrame),
        rawPoints: {
            wrist,
            thumb: handLandmarks[4],
            index: handLandmarks[8],
            middle: handLandmarks[12],
            pinky: handLandmarks[20]
        }
    };

    return { handsVisible, sample };
}

function emptyCalibration() {
    return {
        version: 2,
        wristBody: null,
        points: {},
        capturedAt: null
    };
}

function loadCalibration() {
    try {
        const raw = window.localStorage.getItem(CALIBRATION_STORAGE_KEY);
        if (!raw) {
            return null;
        }
        const parsed = JSON.parse(raw);
        if (!parsed || parsed.version !== 2) {
            return null;
        }
        return parsed;
    } catch (_error) {
        return null;
    }
}

function saveCalibration(calibration) {
    window.localStorage.setItem(CALIBRATION_STORAGE_KEY, JSON.stringify(calibration));
    savedCalibration = calibration;
}

function clearCalibration() {
    window.localStorage.removeItem(CALIBRATION_STORAGE_KEY);
    savedCalibration = null;
}

function getCurrentSyncStep() {
    return syncSession ? SYNC_STEPS[syncSession.stepIndex] || null : null;
}

function startSynchronization() {
    syncSession = {
        stepIndex: 0,
        calibration: emptyCalibration()
    };
    latestPointScores = null;
    holdStartedAt = 0;
    renderCalibrationPanel();
    renderBreakdown(null);
    renderStatus(0, 0, "Synchronization started. Capture the wrist first.");
    renderDiagnostics();
}

function finishSynchronization() {
    syncSession.calibration.capturedAt = new Date().toISOString();
    saveCalibration(syncSession.calibration);
    syncSession = null;
    holdStartedAt = 0;
    renderCalibrationPanel();
    renderStatus(0, 0, "Synchronization saved. Now show A and hold the calibrated pose.");
    renderDiagnostics();
}

function captureCurrentStep() {
    if (!syncSession) {
        startSynchronization();
        return;
    }
    if (!latestSample) {
        renderStatus(0, 0, "Show one clear hand before capturing the current point.");
        return;
    }

    const step = getCurrentSyncStep();
    if (!step) {
        return;
    }

    if (step.id === "wrist") {
        syncSession.calibration.wristBody = clonePoint(latestSample.wristBody);
    } else {
        syncSession.calibration.points[step.id] = clonePoint(latestSample[step.id]);
    }

    syncSession.stepIndex += 1;
    if (syncSession.stepIndex >= SYNC_STEPS.length) {
        finishSynchronization();
        return;
    }

    const nextStep = getCurrentSyncStep();
    renderCalibrationPanel();
    renderStatus(0, 0, `Saved ${step.id}. Next: ${nextStep.title.toLowerCase()}.`);
    renderDiagnostics();
}

function resetSynchronization() {
    syncSession = null;
    clearCalibration();
    latestPointScores = null;
    holdStartedAt = 0;
    renderCalibrationPanel();
    renderBreakdown(null);
    renderStatus(0, 0, "Saved synchronization cleared.");
    renderDiagnostics();
}

function compareWithCalibration(sample) {
    if (!savedCalibration || !savedCalibration.wristBody) {
        return null;
    }

    const pointScores = {
        wrist: scoreDistance(sample.wristBody, savedCalibration.wristBody, POINT_TOLERANCES.wrist),
        thumb: scoreDistance(sample.thumb, savedCalibration.points.thumb, POINT_TOLERANCES.thumb),
        index: scoreDistance(sample.index, savedCalibration.points.index, POINT_TOLERANCES.index),
        middle: scoreDistance(sample.middle, savedCalibration.points.middle, POINT_TOLERANCES.middle),
        pinky: scoreDistance(sample.pinky, savedCalibration.points.pinky, POINT_TOLERANCES.pinky)
    };

    return {
        pointScores,
        score: (
            pointScores.wrist +
            pointScores.thumb +
            pointScores.index +
            pointScores.middle +
            pointScores.pinky
        ) / 5
    };
}

function renderBreakdown(pointScores) {
    scoreBreakdown.innerHTML = "";
    const rows = pointScores
        ? [
            { label: "Wrist anchor", value: pointScores.wrist },
            { label: "Thumb point", value: pointScores.thumb },
            { label: "Index point", value: pointScores.index },
            { label: "Middle point", value: pointScores.middle },
            { label: "Pinky point", value: pointScores.pinky }
        ]
        : [
            { label: "Wrist anchor", value: null },
            { label: "Thumb point", value: null },
            { label: "Index point", value: null },
            { label: "Middle point", value: null },
            { label: "Pinky point", value: null }
        ];

    rows.forEach(rowData => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const hasValue = typeof rowData.value === "number";
        const percent = hasValue ? Math.round(rowData.value * 100) : null;
        const tone = !hasValue ? "is-neutral" : percent >= 70 ? "is-good" : percent >= 40 ? "is-neutral" : "is-bad";
        const valueText = hasValue ? `${percent}%` : "Not calibrated";
        row.innerHTML = `<span>${rowData.label}</span><span class="gesture-check-badge ${tone}">${valueText}</span>`;
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

function renderCalibrationPanel() {
    const currentStep = getCurrentSyncStep();
    const hasCalibration = !!savedCalibration?.wristBody;

    if (currentStep) {
        syncStepBadge.textContent = `Step ${syncSession.stepIndex + 1} / ${SYNC_STEPS.length}`;
        syncStepTitle.textContent = currentStep.title;
        syncStepCopy.textContent = currentStep.description;
        calibrationStatus.textContent = "Synchronization is active. Capture each point in order to create your personal A template.";
        startSyncBtn.textContent = "Restart Sync";
        captureSyncBtn.disabled = false;
    } else if (hasCalibration) {
        syncStepBadge.textContent = "Saved";
        syncStepTitle.textContent = "Synchronization complete";
        syncStepCopy.textContent = "Your five A coordinates are saved. Show A again and the page will compare live points to your calibrated template.";
        calibrationStatus.textContent = "Direct coordinate matching is active for wrist, thumb, index, middle, and pinky.";
        startSyncBtn.textContent = "Run Sync Again";
        captureSyncBtn.disabled = true;
    } else {
        syncStepBadge.textContent = "Not started";
        syncStepTitle.textContent = "Run the five-point synchronization";
        syncStepCopy.textContent = "The page will ask for wrist, thumb, index, middle, and pinky. After that it will only compare live coordinates to your saved A pose.";
        calibrationStatus.textContent = "No saved synchronization yet. Start once, capture the five points, then the score will use only your calibrated coordinates.";
        startSyncBtn.textContent = "Start Synchronization";
        captureSyncBtn.disabled = true;
    }
}

function renderDiagnostics() {
    diagnosticsList.innerHTML = "";

    const currentStep = getCurrentSyncStep();
    const rows = [
        {
            label: "Matcher",
            value: "Pure coordinate matching from your synchronized wrist, thumb, index, middle, and pinky points.",
            badge: "Coords",
            tone: "is-good"
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
        },
        {
            label: "Synchronization",
            value: currentStep
                ? `Active step: ${currentStep.title}.`
                : savedCalibration?.wristBody
                    ? "Saved calibration is active for letter A."
                    : "Calibration has not been created yet.",
            badge: currentStep ? "Active" : savedCalibration?.wristBody ? "Saved" : "Missing",
            tone: currentStep || savedCalibration?.wristBody ? "is-good" : "is-neutral"
        }
    ];

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

    diagnosticsSummary.textContent = currentStep
        ? `Synchronization step ${syncSession.stepIndex + 1} of ${SYNC_STEPS.length}: ${currentStep.title}.`
        : savedCalibration?.wristBody
            ? "Saved A synchronization is active. The live pose is scored only against your saved coordinates."
            : "Start synchronization to save the five A points, then the page will score only against them.";
}

function drawTracking(results) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;

    const currentStep = getCurrentSyncStep();

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    const { handLandmarks } = getPrimaryHand(results);
    if (handLandmarks) {
        CANVAS_POINT_ORDER.forEach(({ id, landmarkIndex }) => {
            const point = handLandmarks[landmarkIndex];
            if (!point) {
                return;
            }
            const isTarget = currentStep?.id === id;
            canvasCtx.beginPath();
            canvasCtx.fillStyle = isTarget ? "rgba(248, 113, 113, 0.98)" : id === "wrist" ? "rgba(251, 191, 36, 0.98)" : "rgba(56, 189, 248, 0.98)";
            canvasCtx.arc(point.x * outputCanvas.width, point.y * outputCanvas.height, isTarget ? 10 : id === "wrist" ? 8 : 7, 0, Math.PI * 2);
            canvasCtx.fill();
        });
    }

    canvasCtx.restore();
}

function handleLiveFrame(results) {
    const { handsVisible, sample } = buildLiveSample(results);
    trackedHands = handsVisible;
    latestSample = sample;
    renderDiagnostics();

    if (!sample) {
        latestPointScores = null;
        renderBreakdown(null);
        holdStartedAt = 0;
        renderStatus(0, 0, syncSession ? "Show one clear hand and capture the requested point." : "Run synchronization, then show A.");
        return;
    }

    if (syncSession) {
        latestPointScores = null;
        renderBreakdown(null);
        holdStartedAt = 0;
        renderStatus(0, 0, `${getCurrentSyncStep().title}. When it looks stable, press Capture Current Point.`);
        return;
    }

    if (!savedCalibration?.wristBody) {
        latestPointScores = null;
        renderBreakdown(null);
        holdStartedAt = 0;
        renderStatus(0, 0, "Start synchronization first so the page can learn your A coordinates.");
        return;
    }

    const comparison = compareWithCalibration(sample);
    latestPointScores = comparison.pointScores;
    renderBreakdown(comparison.pointScores);

    let holdProgress = 0;
    let statusText = "Adjust A until the live points match your synchronized coordinates.";

    if (comparison.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        statusText = elapsed >= HOLD_SECONDS
            ? "A matched your synchronized pose."
            : "Good match. Keep holding the calibrated A.";
    } else {
        holdStartedAt = 0;
    }

    renderStatus(comparison.score, holdProgress, statusText);
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

startSyncBtn.addEventListener("click", startSynchronization);
captureSyncBtn.addEventListener("click", captureCurrentStep);
resetSyncBtn.addEventListener("click", resetSynchronization);
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => refreshPermissionState().catch(console.error));
window.addEventListener("beforeunload", () => stopLoop());

renderCalibrationPanel();
renderBreakdown(null);
renderDiagnostics();
refreshPermissionState().then(() => startCamera()).catch(console.error);
