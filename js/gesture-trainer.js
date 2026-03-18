const MAX_SEQUENCE = 40;
const PREDICTION_INTERVAL = 6;
const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.45;
const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];
const MODEL_HEALTH_URL = "/api/health";
const MODEL_PREDICT_URL = "/api/predict";
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
let diagnosticsPollId = 0;
let lastModelInfo = null;
let lastPredictions = [];
let cameraReady = false;
let detectedHands = 0;

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

function prettifyLabel(label) {
    return label
        .replace(/[0-9]+$/g, "")
        .split(/[_/]/)
        .filter(Boolean)
        .map(chunk => chunk.charAt(0) + chunk.slice(1).toLowerCase())
        .join(" / ");
}

function getCurrentGesture() {
    return gestures[currentGestureIndex] || gestures[0];
}

function setGesture(index) {
    currentGestureIndex = (index + gestures.length) % gestures.length;
    holdStartedAt = 0;
    lastSuccess = false;
    renderGestureCard();
    renderPredictions(lastPredictions);
    renderStatus(0, 0, "Show the trained sign to the camera.");
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

    const targetId = getCurrentGesture().id;
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${prettifyLabel(item.label)}`;
        const value = document.createElement("span");
        value.textContent = `${Math.round(item.score * 100)}%`;
        const isTarget = item.label === targetId;
        const isTop = index === 0;
        value.className = `gesture-check-badge ${isTarget ? "is-good" : isTop ? "is-neutral" : "is-neutral"}`;
        row.append(label, value);
        fingerChecklist.appendChild(row);
    });
}

function setDiagnosticsSummary(text) {
    diagnosticsSummary.textContent = text;
}

function formatPermissionState(state) {
    if (state === "granted") { return "Allowed"; }
    if (state === "denied") { return "Blocked"; }
    if (state === "prompt") { return "Waiting"; }
    return "Unavailable";
}

function renderDiagnosticsRows(rows) {
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

function stopCameraStream() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = 0;
    }
    if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
    }
    cameraReady = false;
}

function describeCameraError(error) {
    if (!error) { return "Camera access failed."; }
    if (error.name === "NotAllowedError") { return "Camera permission was blocked. Allow access in the browser and try again."; }
    if (error.name === "NotFoundError") { return "No camera was found on this device."; }
    if (error.name === "NotReadableError") { return "The camera is busy in another app. Close Zoom, Teams, OBS, or the Windows Camera app, then try again."; }
    if (error.name === "OverconstrainedError") { return "The requested camera settings are not supported on this device."; }
    return `Camera error: ${error.message || error.name}`;
}

async function getCameraPermissionState() {
    if (!navigator.permissions?.query) {
        return "unsupported";
    }
    try {
        const result = await navigator.permissions.query({ name: "camera" });
        return result.state || "unsupported";
    } catch (_error) {
        return "unsupported";
    }
}

async function collectDiagnostics() {
    const support = !!navigator.mediaDevices?.getUserMedia;
    const secureContext = window.isSecureContext;
    const permissionState = await getCameraPermissionState();
    let devices = [];
    let devicesError = "";

    if (navigator.mediaDevices?.enumerateDevices) {
        try {
            devices = await navigator.mediaDevices.enumerateDevices();
        } catch (error) {
            devicesError = error.message || "Could not enumerate media devices.";
        }
    }

    const videoDevices = devices.filter(device => device.kind === "videoinput");
    const activeVideoTrack = activeStream?.getVideoTracks?.()[0] || null;
    const streamActive = !!activeVideoTrack && activeVideoTrack.readyState === "live";
    const selectedLabel = activeVideoTrack?.label || videoDevices[0]?.label || "Not available yet";
    const summaryText =
        !lastModelInfo ? "The page is waiting for the localhost model API." :
        !support ? "This browser does not support camera access on this page." :
        !secureContext ? "Camera access may fail because the page is not in a secure context." :
        permissionState === "denied" ? "Browser permission is blocking the camera for this page." :
        !videoDevices.length ? "No camera devices were detected by the browser." :
        streamActive ? "Camera stream is active and the trained sign model is ready." :
        lastCameraError ? `Camera start failed: ${lastCameraError}` :
        "The model is ready, but the page is still waiting for a live camera stream.";

    setDiagnosticsSummary(summaryText);
    renderDiagnosticsRows([
        {
            label: "Model API",
            value: lastModelInfo ? `Loaded ${lastModelInfo.num_classes} trained classes from ${lastModelInfo.model_name}.` : "The localhost model API has not replied yet.",
            badge: lastModelInfo ? "Ready" : "Offline",
            tone: lastModelInfo ? "is-good" : "is-bad"
        },
        {
            label: "Trained labels",
            value: gestures.length ? `${gestures.length} signs available for practice.` : "No trained labels loaded yet.",
            badge: `${gestures.length}`,
            tone: gestures.length ? "is-good" : "is-neutral"
        },
        {
            label: "Browser camera API",
            value: support ? "navigator.mediaDevices.getUserMedia is available." : "getUserMedia is missing in this browser.",
            badge: support ? "Ready" : "Missing",
            tone: support ? "is-good" : "is-bad"
        },
        {
            label: "Browser permission",
            value: permissionState === "unsupported" ? "Permissions API is unavailable here, so check site settings manually." : `Camera permission state: ${formatPermissionState(permissionState)}.`,
            badge: formatPermissionState(permissionState),
            tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral"
        },
        {
            label: "Detected cameras",
            value: devicesError ? devicesError : videoDevices.length ? videoDevices.map(device => device.label || `Camera ${device.deviceId.slice(0, 8)}`).join(" | ") : "No video input devices reported by the browser.",
            badge: videoDevices.length ? `${videoDevices.length} found` : "0 found",
            tone: videoDevices.length ? "is-good" : "is-bad"
        },
        {
            label: "Active stream",
            value: streamActive ? `Streaming from: ${selectedLabel}` : lastCameraError || "No live video track yet.",
            badge: streamActive ? "Live" : "Idle",
            tone: streamActive ? "is-good" : "is-neutral"
        },
        {
            label: "Tracked signer",
            value: detectedHands ? `Hands detected: ${detectedHands}. Keep one signer centered with upper body visible.` : "No hands are currently visible in the frame.",
            badge: `${detectedHands} hands`,
            tone: detectedHands ? "is-good" : "is-neutral"
        },
        {
            label: "Frame buffer",
            value: `${frameBuffer.length} / ${MAX_SEQUENCE} frames buffered for the next inference window.`,
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

function evaluatePredictionWindow(predictions) {
    const target = getCurrentGesture();
    const top = predictions[0] || null;
    const targetPrediction = predictions.find(item => item.label === target.id) || null;
    const targetScore = targetPrediction?.score || 0;

    let holdProgress = 0;
    let stableSuccess = false;
    let statusText = "Hold the target sign steady so the model can see a full 40-frame sequence.";

    if (top && top.label === target.id && top.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        stableSuccess = elapsed >= HOLD_SECONDS;
        statusText = stableSuccess
            ? `Matched ${target.title}. Moving to the next trained sign.`
            : `Matched ${target.title}. Keep holding it a little longer.`;
    } else if (top) {
        holdStartedAt = 0;
        statusText = `Top guess is ${prettifyLabel(top.label)} at ${Math.round(top.score * 100)}%. Adjust toward ${target.title}.`;
    } else {
        holdStartedAt = 0;
    }

    if (stableSuccess && !lastSuccess) {
        renderStatus(targetScore, 1, `Great job. ${target.title} was recognized.`);
        lastSuccess = true;
        window.setTimeout(() => setGesture(currentGestureIndex + 1), 450);
        return;
    }

    lastSuccess = stableSuccess;
    renderStatus(targetScore, holdProgress, statusText);
}

async function predictSequence() {
    if (predictionInFlight || frameBuffer.length < MAX_SEQUENCE || !lastModelInfo) {
        return;
    }
    predictionInFlight = true;
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
        lastPredictions = result.predictions || [];
        renderPredictions(lastPredictions);
        evaluatePredictionWindow(lastPredictions);
    } catch (error) {
        console.error(error);
        renderStatus(0, 0, `Prediction failed: ${error.message}`);
    } finally {
        predictionInFlight = false;
        collectDiagnostics().catch(console.error);
    }
}

async function checkModel() {
    calibrationStatus.textContent = "Checking localhost model API...";
    calibrateBtn.textContent = "Retry API";
    try {
        const response = await fetch(MODEL_HEALTH_URL);
        if (!response.ok) {
            throw new Error(`Health check failed with status ${response.status}`);
        }
        lastModelInfo = await response.json();
        const labels = lastModelInfo.label_names?.length ? lastModelInfo.label_names : FALLBACK_LABELS;
        gestures = buildGestures(labels);
        currentGestureIndex = Math.min(currentGestureIndex, gestures.length - 1);
        renderGestureCard();
        calibrationStatus.textContent = `Model API ready. Loaded ${gestures.length} trained signs from ${lastModelInfo.model_name}.`;
    } catch (error) {
        console.error(error);
        lastModelInfo = null;
        calibrationStatus.textContent = `Model API unavailable: ${error.message}`;
    }
    await collectDiagnostics();
}

function drawAndQueue(results) {
    drawResults(results);
    appendFrame(results);
    detectedHands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % PREDICTION_INTERVAL === 0) {
        predictSequence().catch(console.error);
    }
}

function startFrameLoop() {
    const processFrame = async () => {
        if (!activeStream || inputVideo.readyState < 2) {
            animationFrameId = requestAnimationFrame(processFrame);
            return;
        }
        try {
            await holistic.send({ image: inputVideo });
        } finally {
            animationFrameId = requestAnimationFrame(processFrame);
        }
    };
    animationFrameId = requestAnimationFrame(processFrame);
}

async function startCamera() {
    stopCameraStream();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(0, 0, "Waiting for camera access.");
    await collectDiagnostics();

    let videoDevices = [];
    if (navigator.mediaDevices?.enumerateDevices) {
        try {
            videoDevices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "videoinput");
        } catch (error) {
            console.warn("Could not enumerate video devices before camera start.", error);
        }
    }

    const attempts = [
        {
            audio: false,
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        },
        {
            audio: false,
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        },
        {
            audio: false,
            video: true
        }
    ];
    if (videoDevices[0]?.deviceId) {
        attempts.unshift({
            audio: false,
            video: {
                deviceId: { exact: videoDevices[0].deviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });
    }

    try {
        let lastError = null;
        for (const constraints of attempts) {
            try {
                activeStream = await navigator.mediaDevices.getUserMedia(constraints);
                break;
            } catch (error) {
                lastError = error;
                if (error?.name !== "NotFoundError" && error?.name !== "OverconstrainedError") {
                    throw error;
                }
            }
        }
        if (!activeStream) {
            throw lastError || new Error("Camera access failed.");
        }
        inputVideo.srcObject = activeStream;
        await inputVideo.play();
        startFrameLoop();
    } catch (error) {
        console.error(error);
        lastCameraError = describeCameraError(error);
        stopCameraStream();
        cameraState.textContent = "Camera access failed";
        renderStatus(0, 0, lastCameraError);
    }
    await collectDiagnostics();
}

async function initTrainer() {
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, "Show the trained sign to the camera.");
    calibrateBtn.textContent = "Retry API";
    setDiagnosticsSummary("Checking model and camera environment...");

    holistic = new window.Holistic({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    holistic.onResults(drawAndQueue);

    if (!navigator.mediaDevices?.getUserMedia) {
        lastCameraError = "This browser does not support getUserMedia on this page.";
        cameraState.textContent = "Camera not supported";
        renderStatus(0, 0, lastCameraError);
        await collectDiagnostics();
        return;
    }

    await checkModel();
    await startCamera();

    if (navigator.mediaDevices?.addEventListener) {
        navigator.mediaDevices.addEventListener("devicechange", () => {
            collectDiagnostics().catch(console.error);
        });
    }

    if (diagnosticsPollId) {
        clearInterval(diagnosticsPollId);
    }
    diagnosticsPollId = window.setInterval(() => {
        collectDiagnostics().catch(console.error);
    }, 3000);
}

nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
calibrateBtn.addEventListener("click", () => checkModel().catch(console.error));

window.addEventListener("beforeunload", () => {
    if (diagnosticsPollId) {
        clearInterval(diagnosticsPollId);
    }
    stopCameraStream();
});

initTrainer().catch(error => {
    console.error(error);
    cameraState.textContent = "Trainer failed to load";
    renderStatus(0, 0, "The page could not initialize the trained sign pipeline.");
});
