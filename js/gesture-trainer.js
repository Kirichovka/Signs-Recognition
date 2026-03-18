const GESTURES = [
    { id: "open_palm", title: "Open Palm", badge: "OPEN", instruction: "Open your hand and stretch all fingers.", fingers: { thumb: "open", index: "open", middle: "open", ring: "open", pinky: "open" } },
    { id: "closed_fist", title: "Closed Fist", badge: "FIST", instruction: "Make a gentle fist.", fingers: { thumb: "closed", index: "closed", middle: "closed", ring: "closed", pinky: "closed" } },
    { id: "point_up", title: "Point Up", badge: "POINT", instruction: "Lift your index finger and fold the other fingers.", fingers: { thumb: "closed", index: "open", middle: "closed", ring: "closed", pinky: "closed" } },
    { id: "victory", title: "Victory", badge: "V", instruction: "Open your index and middle fingers.", fingers: { thumb: "closed", index: "open", middle: "open", ring: "closed", pinky: "closed" } },
    { id: "thumbs_up", title: "Thumbs Up", badge: "UP", instruction: "Raise your thumb and keep the other fingers closed.", fingers: { thumb: "open", index: "closed", middle: "closed", ring: "closed", pinky: "closed" } }
];

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.74;
const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];
const CALIBRATION_SAMPLE_TARGET = 18;

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

let currentGestureIndex = 0;
let holdStartedAt = 0;
let lastSuccess = false;
let activeStream = null;
let animationFrameId = 0;
let handsInstance = null;
let lastCameraError = "";
let diagnosticsPollId = 0;
let detectedHandCount = 0;

const calibrationState = {
    running: false,
    samples: [],
    profile: null
};

function getCurrentGesture() {
    return GESTURES[currentGestureIndex];
}

function setGesture(index) {
    currentGestureIndex = (index + GESTURES.length) % GESTURES.length;
    holdStartedAt = 0;
    lastSuccess = false;
    renderGestureCard();
    renderEvaluation(null, 0, "Show one or two hands to the camera.");
}

function renderGestureCard() {
    const gesture = getCurrentGesture();
    gestureTitle.textContent = gesture.title;
    gestureInstruction.textContent = gesture.instruction;
    gestureEmoji.textContent = gesture.badge;
    gestureStep.textContent = `${currentGestureIndex + 1} / ${GESTURES.length}`;
}

function renderChecklist(result) {
    fingerChecklist.innerHTML = "";
    for (const fingerName of FINGER_NAMES) {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = fingerName[0].toUpperCase() + fingerName.slice(1);
        const value = document.createElement("span");
        if (!result) {
            value.textContent = "Waiting";
            value.className = "gesture-check-badge is-neutral";
        } else {
            const observed = result.observed[fingerName];
            const expected = result.expected[fingerName];
            value.textContent = `${observed} / ${expected}`;
            value.className = `gesture-check-badge ${result.matches[fingerName] ? "is-good" : "is-bad"}`;
        }
        row.append(label, value);
        fingerChecklist.appendChild(row);
    }
}

function renderEvaluation(result, holdProgress, statusText) {
    const scoreValue = result ? Math.round(result.score * 100) : 0;
    gestureScore.textContent = `${scoreValue}%`;
    gestureStatus.textContent = statusText;
    holdProgressBar.style.width = `${Math.round(holdProgress * 100)}%`;
    holdProgressValue.textContent = `${Math.round(holdProgress * 100)}%`;
    gestureScore.classList.toggle("is-strong", scoreValue >= 80);
    renderChecklist(result);
}

function renderCalibrationStatus(text) {
    calibrationStatus.textContent = text;
}

function syncCalibrationButton() {
    calibrateBtn.textContent = calibrationState.running ? "Cancel" : "Calibrate";
    calibrateBtn.classList.toggle("secondary", calibrationState.running);
}

function setDiagnosticsSummary(text) {
    diagnosticsSummary.textContent = text;
}

function formatPermissionState(state) {
    if (state === "granted") { return "Allowed"; }
    if (state === "denied") { return "Blocked"; }
    if (state === "prompt") { return "Waiting for browser permission"; }
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

function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleBetween(a, b, c) {
    const abx = a.x - b.x;
    const aby = a.y - b.y;
    const cbx = c.x - b.x;
    const cby = c.y - b.y;
    const dot = abx * cbx + aby * cby;
    const magAb = Math.hypot(abx, aby);
    const magCb = Math.hypot(cbx, cby);
    if (!magAb || !magCb) {
        return 180;
    }
    const cosine = Math.max(-1, Math.min(1, dot / (magAb * magCb)));
    return Math.acos(cosine) * (180 / Math.PI);
}

function getFingerMap() {
    return {
        thumb: [1, 2, 3, 4],
        index: [5, 6, 7, 8],
        middle: [9, 10, 11, 12],
        ring: [13, 14, 15, 16],
        pinky: [17, 18, 19, 20]
    };
}

function getHandScale(landmarks) {
    const wrist = landmarks[0];
    const middleMcp = landmarks[9];
    const indexMcp = landmarks[5];
    const pinkyMcp = landmarks[17];
    return (distance(wrist, middleMcp) + distance(indexMcp, pinkyMcp)) / 2;
}

function getFingerMetrics(landmarks, handednessLabel) {
    const wrist = landmarks[0];
    const fingerMap = getFingerMap();
    const metrics = {};
    for (const [fingerName, [mcp, pip, dip, tip]] of Object.entries(fingerMap)) {
        const pipAngle = angleBetween(landmarks[mcp], landmarks[pip], landmarks[dip]);
        const dipAngle = angleBetween(landmarks[pip], landmarks[dip], landmarks[tip]);
        const tipToWrist = distance(landmarks[tip], wrist);
        const pipToWrist = distance(landmarks[pip], wrist);
        const ratio = pipToWrist ? tipToWrist / pipToWrist : 1;
        const thumbDirection = handednessLabel === "Right"
            ? landmarks[tip].x < landmarks[3].x && landmarks[tip].x < landmarks[mcp].x
            : landmarks[tip].x > landmarks[3].x && landmarks[tip].x > landmarks[mcp].x;
        metrics[fingerName] = {
            pipAngle,
            dipAngle,
            tipToWrist,
            pipToWrist,
            ratio,
            thumbDirection
        };
    }
    return metrics;
}

function average(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function buildCalibrationProfile(samples) {
    const fingers = {};
    for (const fingerName of FINGER_NAMES) {
        fingers[fingerName] = {
            ratio: average(samples.map(sample => sample.fingers[fingerName].ratio)),
            pipAngle: average(samples.map(sample => sample.fingers[fingerName].pipAngle)),
            dipAngle: average(samples.map(sample => sample.fingers[fingerName].dipAngle))
        };
    }
    return {
        createdAt: Date.now(),
        handScale: average(samples.map(sample => sample.handScale)),
        fingers
    };
}

function getAdaptiveThresholds(fingerName, currentHandScale) {
    const profile = calibrationState.profile;
    const currentScale = currentHandScale || 0.18;
    const relativeScale = profile?.handScale ? currentScale / profile.handScale : 1;
    const smallHandBoost = relativeScale < 0.9 ? (0.9 - relativeScale) : 0;
    if (fingerName === "thumb") {
        const baseRatio = profile?.fingers.thumb.ratio ?? 1.2;
        const basePip = profile?.fingers.thumb.pipAngle ?? 160;
        return {
            ratio: Math.max(1.02, baseRatio * (0.74 - smallHandBoost * 0.12)),
            pipAngle: Math.max(130, basePip - 28 - smallHandBoost * 10)
        };
    }
    const baseRatio = profile?.fingers[fingerName].ratio ?? 1.22;
    const basePip = profile?.fingers[fingerName].pipAngle ?? 172;
    const baseDip = profile?.fingers[fingerName].dipAngle ?? 166;
    return {
        ratio: Math.max(1.04, baseRatio * (0.8 - smallHandBoost * 0.12)),
        pipAngle: Math.max(138, basePip - 24 - smallHandBoost * 10),
        dipAngle: Math.max(132, baseDip - 22 - smallHandBoost * 10)
    };
}

function detectFingerStates(landmarks, handednessLabel) {
    const metrics = getFingerMetrics(landmarks, handednessLabel);
    const currentHandScale = getHandScale(landmarks);
    const states = {};

    for (const fingerName of FINGER_NAMES) {
        const fingerMetrics = metrics[fingerName];
        const thresholds = getAdaptiveThresholds(fingerName, currentHandScale);
        if (fingerName === "thumb") {
            const isOpen = fingerMetrics.thumbDirection
                || fingerMetrics.ratio >= thresholds.ratio
                || fingerMetrics.pipAngle >= thresholds.pipAngle;
            states[fingerName] = isOpen ? "open" : "closed";
            continue;
        }
        const straightEnough = fingerMetrics.pipAngle >= thresholds.pipAngle && fingerMetrics.dipAngle >= thresholds.dipAngle;
        const farEnough = fingerMetrics.ratio >= thresholds.ratio;
        states[fingerName] = straightEnough && farEnough ? "open" : "closed";
    }

    return { states, metrics, handScale: currentHandScale };
}

function evaluateGesture(observed, expected) {
    let passedChecks = 0;
    let totalChecks = 0;
    const matches = {};
    for (const fingerName of FINGER_NAMES) {
        const matched = expected[fingerName] === observed[fingerName];
        matches[fingerName] = matched;
        totalChecks += 1;
        if (matched) {
            passedChecks += 1;
        }
    }
    return {
        score: totalChecks ? passedChecks / totalChecks : 0,
        matches,
        expected,
        observed
    };
}

function getBestHandEvaluation(results) {
    if (!results.multiHandLandmarks?.length || !results.multiHandedness?.length) {
        return null;
    }

    const gesture = getCurrentGesture();
    const evaluations = results.multiHandLandmarks.map((landmarks, index) => {
        const handedness = results.multiHandedness[index]?.label || "Unknown";
        const detection = detectFingerStates(landmarks, handedness);
        const evaluation = evaluateGesture(detection.states, gesture.fingers);
        return {
            evaluation,
            handedness,
            metrics: detection.metrics,
            handScale: detection.handScale
        };
    });

    return evaluations.sort((a, b) => b.evaluation.score - a.evaluation.score)[0] || null;
}

function drawResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);
    if (results.multiHandLandmarks) {
        results.multiHandLandmarks.forEach((landmarks, index) => {
            const handedness = results.multiHandedness?.[index]?.label;
            const strokeColor = handedness === "Left" ? "#f97316" : "#38bdf8";
            const fillColor = handedness === "Left" ? "#7c2d12" : "#1d4ed8";
            window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, { color: strokeColor, lineWidth: 4 });
            window.drawLandmarks(canvasCtx, landmarks, { color: "#eff6ff", fillColor, radius: 4 });
        });
    }
    canvasCtx.restore();
}

function stopCameraStream() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = 0;
    }
    if (activeStream) {
        for (const track of activeStream.getTracks()) {
            track.stop();
        }
        activeStream = null;
    }
}

function describeCameraError(error) {
    if (!error) { return "Camera access failed."; }
    if (error.name === "NotAllowedError") { return "Camera permission was blocked. Allow access in the browser and try again."; }
    if (error.name === "NotFoundError") { return "No camera was found on this device."; }
    if (error.name === "NotReadableError") { return "The camera is busy in another app. Close Zoom, Teams, OBS, or the Windows Camera app, then try again."; }
    if (error.name === "OverconstrainedError") { return "The requested camera settings are not supported on this device."; }
    return `Camera error: ${error.message || error.name}`;
}

function processCalibrationFrame(results) {
    if (!calibrationState.running) {
        return;
    }
    if (!results.multiHandLandmarks?.length || !results.multiHandedness?.length) {
        renderCalibrationStatus("Calibration paused. Show one open palm clearly to the camera.");
        return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const handedness = results.multiHandedness[0]?.label || "Unknown";
    const detection = detectFingerStates(landmarks, handedness);
    const openPalmScore = evaluateGesture(detection.states, GESTURES[0].fingers).score;

    if (openPalmScore < 0.8) {
        renderCalibrationStatus("Calibration needs an open palm. Stretch your fingers and hold steady.");
        return;
    }

    calibrationState.samples.push({
        handScale: detection.handScale,
        fingers: detection.metrics
    });

    if (calibrationState.samples.length >= CALIBRATION_SAMPLE_TARGET) {
        calibrationState.profile = buildCalibrationProfile(calibrationState.samples);
        calibrationState.running = false;
        syncCalibrationButton();
        renderCalibrationStatus(`Calibrated from ${CALIBRATION_SAMPLE_TARGET} live samples. Detection is now tuned for your hand size and distance.`);
        collectDiagnostics().catch(console.error);
        return;
    }

    renderCalibrationStatus(`Calibrating open palm: ${calibrationState.samples.length} / ${CALIBRATION_SAMPLE_TARGET} samples collected.`);
}

async function getCameraPermissionState() {
    if (!navigator.permissions?.query) {
        return "unsupported";
    }
    try {
        const result = await navigator.permissions.query({ name: "camera" });
        return result.state || "unsupported";
    } catch (error) {
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
    const videoReady = inputVideo.readyState >= 2;
    const hasDimensions = inputVideo.videoWidth > 0 && inputVideo.videoHeight > 0;

    const summaryText =
        !support ? "This browser does not support camera access on this page." :
        !secureContext ? "Camera access may fail because the page is not in a secure context." :
        permissionState === "denied" ? "Browser permission is blocking the camera for this page." :
        !videoDevices.length ? "No camera devices were detected by the browser." :
        streamActive ? "Camera stream is active and the page can read frames." :
        lastCameraError ? `Camera start failed: ${lastCameraError}` :
        "The page can see camera support, but an active stream has not started yet.";

    setDiagnosticsSummary(summaryText);

    renderDiagnosticsRows([
        {
            label: "Browser camera API",
            value: support ? "navigator.mediaDevices.getUserMedia is available." : "getUserMedia is missing in this browser.",
            badge: support ? "Ready" : "Missing",
            tone: support ? "is-good" : "is-bad"
        },
        {
            label: "Secure context",
            value: secureContext ? "The page is running in a secure context, which is required for camera access." : `Current origin: ${window.location.origin}`,
            badge: secureContext ? "Secure" : "Not secure",
            tone: secureContext ? "is-good" : "is-bad"
        },
        {
            label: "Browser permission",
            value: permissionState === "unsupported" ? "Permissions API is unavailable here, so check the browser site settings manually." : `Camera permission state: ${formatPermissionState(permissionState)}.`,
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
            label: "Tracked hands",
            value: detectedHandCount ? `MediaPipe is currently tracking ${detectedHandCount} hand(s). Two hands are enabled.` : "No hands are currently visible in the frame.",
            badge: `${detectedHandCount} live`,
            tone: detectedHandCount ? "is-good" : "is-neutral"
        },
        {
            label: "Calibration profile",
            value: calibrationState.profile ? "Adaptive thresholds are using your saved live hand measurements." : calibrationState.running ? "Calibration is collecting live open-palm samples." : "No saved calibration yet. Detection uses adaptive defaults.",
            badge: calibrationState.profile ? "Ready" : calibrationState.running ? "Running" : "Default",
            tone: calibrationState.profile ? "is-good" : "is-neutral"
        },
        {
            label: "Active stream",
            value: streamActive ? `Streaming from: ${selectedLabel}` : lastCameraError || "No live video track yet.",
            badge: streamActive ? "Live" : "Idle",
            tone: streamActive ? "is-good" : "is-neutral"
        },
        {
            label: "Video frame state",
            value: `readyState=${inputVideo.readyState}, size=${inputVideo.videoWidth || 0}x${inputVideo.videoHeight || 0}, canvas=${outputCanvas.width || 0}x${outputCanvas.height || 0}`,
            badge: videoReady && hasDimensions ? "Frames ready" : "Waiting",
            tone: videoReady && hasDimensions ? "is-good" : "is-neutral"
        },
        {
            label: "Last camera error",
            value: lastCameraError || "No camera error has been recorded in this session.",
            badge: lastCameraError ? "Has error" : "Clear",
            tone: lastCameraError ? "is-bad" : "is-good"
        },
        {
            label: "Page address",
            value: `${window.location.href}`,
            badge: window.location.protocol === "https:" || window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost" ? "Allowed origin" : "Check origin",
            tone: window.location.protocol === "https:" || window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost" ? "is-good" : "is-neutral"
        }
    ]);
}

function processVideoFrame() {
    if (!handsInstance || !activeStream || inputVideo.readyState < 2) {
        animationFrameId = requestAnimationFrame(processVideoFrame);
        return;
    }
    handsInstance.send({ image: inputVideo }).finally(() => {
        animationFrameId = requestAnimationFrame(processVideoFrame);
    });
}

async function startCamera() {
    stopCameraStream();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    gestureStatus.textContent = "Waiting for camera access.";
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
        cameraState.textContent = "Camera is live";
        gestureStatus.textContent = "Show one or two hands to the camera.";
        await collectDiagnostics();
        processVideoFrame();
    } catch (error) {
        console.error(error);
        lastCameraError = describeCameraError(error);
        stopCameraStream();
        cameraState.textContent = "Camera access failed";
        gestureStatus.textContent = lastCameraError;
        await collectDiagnostics();
    }
}

function beginCalibration() {
    if (calibrationState.running) {
        calibrationState.running = false;
        calibrationState.samples = [];
        syncCalibrationButton();
        renderCalibrationStatus("Calibration canceled. You can start again whenever your hand is centered.");
        collectDiagnostics().catch(console.error);
        return;
    }
    calibrationState.running = true;
    calibrationState.samples = [];
    syncCalibrationButton();
    renderCalibrationStatus("Calibration started. Show one open palm close to your normal play distance and hold still.");
}

async function initTrainer() {
    renderGestureCard();
    renderEvaluation(null, 0, "Show one or two hands to the camera.");
    renderCalibrationStatus("Not calibrated yet. Show an open palm and press Calibrate.");
    syncCalibrationButton();
    setDiagnosticsSummary("Checking camera environment...");

    handsInstance = new window.Hands({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    handsInstance.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.45
    });

    handsInstance.onResults(results => {
        outputCanvas.width = results.image.width;
        outputCanvas.height = results.image.height;
        detectedHandCount = results.multiHandLandmarks?.length || 0;
        drawResults(results);
        processCalibrationFrame(results);

        let statusText = detectedHandCount ? `Tracking ${detectedHandCount} hand(s).` : "Show one or two hands to the camera.";
        let holdProgress = 0;
        let evaluation = null;
        let stableSuccess = false;

        const bestHand = getBestHandEvaluation(results);

        if (bestHand) {
            evaluation = bestHand.evaluation;
            if (evaluation.score >= SCORE_THRESHOLD) {
                if (!holdStartedAt) {
                    holdStartedAt = performance.now();
                }
                const elapsed = (performance.now() - holdStartedAt) / 1000;
                holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
                stableSuccess = elapsed >= HOLD_SECONDS;
                statusText = stableSuccess
                    ? `Great job! ${bestHand.handedness} hand matched.`
                    : `Best match: ${bestHand.handedness} hand. Hold the pose a little longer.`;
            } else {
                holdStartedAt = 0;
                statusText = detectedHandCount > 1
                    ? "Two hands detected. Adjust either hand to match the target sign."
                    : "Adjust your fingers to match the target sign.";
            }
        } else {
            holdStartedAt = 0;
        }

        if (stableSuccess && !lastSuccess) {
            setGesture(currentGestureIndex + 1);
            statusText = "Great job! Moving to the next sign.";
            holdProgress = 1;
        }

        lastSuccess = stableSuccess;
        renderEvaluation(evaluation, holdProgress, statusText);
    });

    if (!navigator.mediaDevices?.getUserMedia) {
        lastCameraError = "This browser does not support getUserMedia on this page.";
        cameraState.textContent = "Camera not supported";
        gestureStatus.textContent = lastCameraError;
        await collectDiagnostics();
        return;
    }

    await collectDiagnostics();

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

    await startCamera();
}

nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => {
    startCamera().catch(error => {
        console.error(error);
        cameraState.textContent = "Camera access failed";
        gestureStatus.textContent = describeCameraError(error);
    });
});
refreshDiagnosticsBtn.addEventListener("click", () => {
    collectDiagnostics().catch(console.error);
});
calibrateBtn.addEventListener("click", () => {
    beginCalibration();
});

window.addEventListener("beforeunload", () => {
    if (diagnosticsPollId) {
        clearInterval(diagnosticsPollId);
    }
    stopCameraStream();
});

initTrainer().catch(error => {
    console.error(error);
    cameraState.textContent = "Trainer failed to load";
    gestureStatus.textContent = "The page could not initialize MediaPipe.";
});
