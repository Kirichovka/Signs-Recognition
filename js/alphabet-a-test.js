import {
    describeCameraError,
    getCameraPermissionState,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.4;
const LETTER_POINTS = [0, 4, 8, 12, 20];
const BOUND_POSE_PREFIX = "gesture-trainer.bound-letter.";

const LETTER_BUTTONS = Array.from(document.querySelectorAll("[data-letter]"));
const letterTitle = document.getElementById("letter-title");
const targetCopy = document.getElementById("target-copy");
const primitiveLabel = document.getElementById("primitive-label");
const gestureScore = document.getElementById("gesture-score");
const gestureStatus = document.getElementById("gesture-status");
const holdProgressBar = document.getElementById("hold-progress-bar");
const holdProgressValue = document.getElementById("hold-progress-value");
const scoreBreakdown = document.getElementById("score-breakdown");
const cameraState = document.getElementById("camera-state");
const retryCameraBtn = document.getElementById("retry-camera-btn");
const refreshDiagnosticsBtn = document.getElementById("refresh-diagnostics-btn");
const bindPoseBtn = document.getElementById("bind-pose-btn");
const resetPoseBtn = document.getElementById("reset-pose-btn");
const diagnosticsSummary = document.getElementById("diagnostics-summary");
const diagnosticsList = document.getElementById("diagnostics-list");
const inputVideo = document.getElementById("input-video");
const outputCanvas = document.getElementById("output-canvas");

const canvasCtx = outputCanvas.getContext("2d");

const PRIMITIVE_TEMPLATES = {
    fist: [[0, 0], [-0.8, -0.15], [-0.35, -0.42], [0, -0.38], [0.55, -0.25]],
    flat: [[0, 0], [-0.88, -0.12], [-0.38, -1.02], [0.02, -1.08], [0.78, -0.96]],
    curve: [[0, 0], [-0.7, -0.35], [-0.42, -0.86], [0.0, -0.92], [0.68, -0.48]],
    lshape: [[0, 0], [-0.98, -0.02], [-0.1, -1.02], [0.12, -0.3], [0.54, -0.12]],
    yshape: [[0, 0], [-0.98, -0.24], [-0.18, -0.16], [0.08, -0.12], [0.9, -0.98]]
};

const PRIMITIVE_VECTORS = Object.fromEntries(
    Object.entries(PRIMITIVE_TEMPLATES).map(([name, points]) => [name, pairwiseVector(points.map(([x, y]) => ({ x, y })))])
);

const LETTER_SPECS = {
    A: {
        primitive: "fist",
        title: "Letter A",
        description: "Closed fist with the thumb resting along the side/front.",
        primitiveHint: "fist",
        scorer: scoreLetterAComponents
    },
    B: {
        primitive: "flat",
        title: "Letter B",
        description: "Flat upright hand with straight fingers and the thumb folded across the palm.",
        primitiveHint: "flat hand",
        scorer: scoreLetterBComponents
    },
    C: {
        primitive: "curve",
        title: "Letter C",
        description: "Hand curved into a clear C-shape with open space between thumb and fingers.",
        primitiveHint: "curve",
        scorer: scoreLetterCComponents
    },
    L: {
        primitive: "lshape",
        title: "Letter L",
        description: "Index finger up, thumb out, other fingers folded in.",
        primitiveHint: "L-shape",
        scorer: scoreLetterLComponents
    },
    Y: {
        primitive: "yshape",
        title: "Letter Y",
        description: "Thumb and pinky extended with the middle fingers folded.",
        primitiveHint: "Y-shape",
        scorer: scoreLetterYComponents
    }
};

let currentLetter = "A";
let activeStream = null;
let animationFrameId = 0;
let holistic = null;
let holdStartedAt = 0;
let cameraReady = false;
let trackedHands = 0;
let lastCameraError = "";
let latestDebug = null;
let latestScoredSample = null;
let boundPose = loadBoundPose(currentLetter);

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

function flattenPoints(points) {
    return points.flatMap(([x, y]) => [x, y]);
}

function compareVectors(left, right, tolerance) {
    const diff = left.reduce((sum, value, index) => sum + Math.abs(value - right[index]), 0) / left.length;
    return Math.max(0, 1 - diff / tolerance);
}

function scoreProximity(actual, target, tolerance) {
    return clamp01(1 - Math.abs(actual - target) / tolerance);
}

function angleDegrees(left, pivot, right) {
    const leftVectorX = left.x - pivot.x;
    const leftVectorY = left.y - pivot.y;
    const rightVectorX = right.x - pivot.x;
    const rightVectorY = right.y - pivot.y;
    const leftLength = Math.hypot(leftVectorX, leftVectorY);
    const rightLength = Math.hypot(rightVectorX, rightVectorY);
    if (!leftLength || !rightLength) {
        return 180;
    }
    const dot = (leftVectorX * rightVectorX) + (leftVectorY * rightVectorY);
    const cosine = Math.max(-1, Math.min(1, dot / (leftLength * rightLength)));
    return Math.acos(cosine) * (180 / Math.PI);
}

function scoreCurledFinger(mcp, pip, dip, tip) {
    const pipAngle = angleDegrees(mcp, pip, dip);
    const dipAngle = angleDegrees(pip, dip, tip);
    return {
        score: (scoreProximity(pipAngle, 82, 58) * 0.6) + (scoreProximity(dipAngle, 96, 62) * 0.4),
        pipAngle,
        dipAngle
    };
}

function scoreStraightFinger(mcp, pip, dip, tip) {
    const pipAngle = angleDegrees(mcp, pip, dip);
    const dipAngle = angleDegrees(pip, dip, tip);
    return {
        score: (scoreProximity(pipAngle, 170, 28) * 0.6) + (scoreProximity(dipAngle, 172, 24) * 0.4),
        pipAngle,
        dipAngle
    };
}

function extractSparseHand(handLandmarks) {
    if (!handLandmarks?.length) {
        return null;
    }
    return LETTER_POINTS.map(index => handLandmarks[index] || handLandmarks[0]);
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

function thumbMetricsInHandFrame(wrist, indexMcp, middleMcp, thumbTip, handScale) {
    const upXRaw = middleMcp.x - wrist.x;
    const upYRaw = middleMcp.y - wrist.y;
    const upLength = Math.hypot(upXRaw, upYRaw) || 1;
    const upX = upXRaw / upLength;
    const upY = upYRaw / upLength;
    const sideX = -upY;
    const sideY = upX;
    const thumbVecX = thumbTip.x - indexMcp.x;
    const thumbVecY = thumbTip.y - indexMcp.y;
    return {
        thumbLateral: Math.abs(((thumbVecX * sideX) + (thumbVecY * sideY)) / handScale),
        thumbForward: Math.abs(((thumbVecX * upX) + (thumbVecY * upY)) / handScale)
    };
}

function getFingerMetrics(handLandmarks) {
    const wrist = handLandmarks[0];
    const thumbTip = handLandmarks[4];
    const indexMcp = handLandmarks[5];
    const indexPip = handLandmarks[6];
    const indexDip = handLandmarks[7];
    const indexTip = handLandmarks[8];
    const middleMcp = handLandmarks[9];
    const middlePip = handLandmarks[10];
    const middleDip = handLandmarks[11];
    const middleTip = handLandmarks[12];
    const ringMcp = handLandmarks[13];
    const ringPip = handLandmarks[14];
    const ringDip = handLandmarks[15];
    const ringTip = handLandmarks[16];
    const pinkyMcp = handLandmarks[17];
    const pinkyPip = handLandmarks[18];
    const pinkyDip = handLandmarks[19];
    const pinkyTip = handLandmarks[20];

    const handScale = Math.max(
        0.02,
        average([
            distance2D(wrist, indexMcp),
            distance2D(wrist, middleMcp),
            distance2D(wrist, ringMcp),
            distance2D(wrist, pinkyMcp)
        ])
    );

    const { thumbLateral, thumbForward } = thumbMetricsInHandFrame(wrist, indexMcp, middleMcp, thumbTip, handScale);

    return {
        wrist,
        thumbTip,
        indexMcp,
        indexStraight: scoreStraightFinger(indexMcp, indexPip, indexDip, indexTip),
        indexCurled: scoreCurledFinger(indexMcp, indexPip, indexDip, indexTip),
        middleStraight: scoreStraightFinger(middleMcp, middlePip, middleDip, middleTip),
        middleCurled: scoreCurledFinger(middleMcp, middlePip, middleDip, middleTip),
        ringStraight: scoreStraightFinger(ringMcp, ringPip, ringDip, ringTip),
        ringCurled: scoreCurledFinger(ringMcp, ringPip, ringDip, ringTip),
        pinkyStraight: scoreStraightFinger(pinkyMcp, pinkyPip, pinkyDip, pinkyTip),
        pinkyCurled: scoreCurledFinger(pinkyMcp, pinkyPip, pinkyDip, pinkyTip),
        thumbLateral,
        thumbForward,
        handScale
    };
}

function primitiveScoreForHand(handLandmarks, primitiveName) {
    const sparse = extractSparseHand(handLandmarks);
    if (!sparse) {
        return { primitiveScore: 0, flattenedSparse: null };
    }
    const normalizedSparse = normalizePoints(sparse);
    const flattenedSparse = flattenPoints(normalizedSparse);
    const primitiveVector = PRIMITIVE_VECTORS[primitiveName];
    return {
        primitiveScore: compareVectors(pairwiseVector(sparse), primitiveVector, 0.2),
        flattenedSparse
    };
}

function scoreLetterAComponents(metrics) {
    return {
        fingerScore: average([
            metrics.indexCurled.score,
            metrics.middleCurled.score,
            metrics.ringCurled.score,
            metrics.pinkyCurled.score
        ]),
        thumbScore: (scoreProximity(metrics.thumbLateral, 0.62, 0.36) * 0.75) + (scoreProximity(metrics.thumbForward, 0.14, 0.24) * 0.25)
    };
}

function scoreLetterBComponents(metrics) {
    return {
        fingerScore: average([
            metrics.indexStraight.score,
            metrics.middleStraight.score,
            metrics.ringStraight.score,
            metrics.pinkyStraight.score
        ]),
        thumbScore: (scoreProximity(metrics.thumbLateral, 0.2, 0.22) * 0.55) + (scoreProximity(metrics.thumbForward, 0.28, 0.26) * 0.45)
    };
}

function scoreLetterCComponents(metrics) {
    return {
        fingerScore: average([
            scoreProximity(metrics.indexStraight.pipAngle, 128, 42),
            scoreProximity(metrics.middleStraight.pipAngle, 132, 42),
            scoreProximity(metrics.ringStraight.pipAngle, 128, 42),
            scoreProximity(metrics.pinkyStraight.pipAngle, 122, 45)
        ]),
        thumbScore: (scoreProximity(metrics.thumbLateral, 0.4, 0.28) * 0.7) + (scoreProximity(metrics.thumbForward, 0.34, 0.24) * 0.3)
    };
}

function scoreLetterLComponents(metrics) {
    return {
        fingerScore: average([
            metrics.indexStraight.score,
            metrics.middleCurled.score,
            metrics.ringCurled.score,
            metrics.pinkyCurled.score
        ]),
        thumbScore: (scoreProximity(metrics.thumbLateral, 0.9, 0.35) * 0.75) + (scoreProximity(metrics.thumbForward, 0.12, 0.22) * 0.25)
    };
}

function scoreLetterYComponents(metrics) {
    return {
        fingerScore: average([
            metrics.indexCurled.score,
            metrics.middleCurled.score,
            metrics.ringCurled.score,
            metrics.pinkyStraight.score
        ]),
        thumbScore: (scoreProximity(metrics.thumbLateral, 0.92, 0.36) * 0.75) + (scoreProximity(metrics.thumbForward, 0.16, 0.24) * 0.25)
    };
}

function boundPoseKey(letter) {
    return `${BOUND_POSE_PREFIX}${letter}`;
}

function loadBoundPose(letter) {
    try {
        const raw = window.localStorage.getItem(boundPoseKey(letter));
        return raw ? JSON.parse(raw) : null;
    } catch (_error) {
        return null;
    }
}

function saveBoundPose(letter, sample) {
    window.localStorage.setItem(boundPoseKey(letter), JSON.stringify(sample));
    boundPose = sample;
}

function clearBoundPose(letter) {
    window.localStorage.removeItem(boundPoseKey(letter));
    boundPose = null;
}

function evaluateBoundPose(sample) {
    if (!boundPose || !sample?.flattenedSparse) {
        return 0;
    }
    const pointScore = compareVectors(sample.flattenedSparse, boundPose.flattenedSparse, 0.6);
    const angleScore = compareVectors(sample.fingerAngles, boundPose.fingerAngles, 70);
    const thumbScore = (scoreProximity(sample.thumbLateral, boundPose.thumbLateral, 0.38) * 0.7) + (scoreProximity(sample.thumbForward, boundPose.thumbForward, 0.32) * 0.3);
    const bodyScore = (scoreProximity(sample.wristX, boundPose.wristX, 0.9) * 0.45) + (scoreProximity(sample.wristY, boundPose.wristY, 0.9) * 0.55);
    return (pointScore * 0.35) + (angleScore * 0.3) + (thumbScore * 0.2) + (bodyScore * 0.15);
}

function scoreCurrentLetter(results) {
    const hands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean);
    if (!hands.length) {
        return { score: 0, debug: null, sample: null, handsVisible: 0 };
    }

    const spec = LETTER_SPECS[currentLetter];
    const bodyFrame = getBodyFrame(results);
    const candidates = hands.map(handLandmarks => {
        const metrics = getFingerMetrics(handLandmarks);
        const { primitiveScore, flattenedSparse } = primitiveScoreForHand(handLandmarks, spec.primitive);
        const { fingerScore, thumbScore } = spec.scorer(metrics);
        const wristX = (metrics.wrist.x - bodyFrame.center.x) / bodyFrame.scale;
        const wristY = (metrics.wrist.y - bodyFrame.center.y) / bodyFrame.scale;
        const bodyScore = (scoreProximity(Math.abs(wristX), 0.55, 0.85) * 0.45) + (scoreProximity(wristY, 0.15, 0.8) * 0.55);
        const sample = {
            flattenedSparse,
            fingerAngles: [
                metrics.indexStraight.pipAngle, metrics.indexStraight.dipAngle,
                metrics.middleStraight.pipAngle, metrics.middleStraight.dipAngle,
                metrics.ringStraight.pipAngle, metrics.ringStraight.dipAngle,
                metrics.pinkyStraight.pipAngle, metrics.pinkyStraight.dipAngle
            ],
            thumbLateral: metrics.thumbLateral,
            thumbForward: metrics.thumbForward,
            wristX,
            wristY
        };
        const templateScore = evaluateBoundPose(sample);
        const defaultScore = (primitiveScore * 0.4) + (fingerScore * 0.35) + (thumbScore * 0.15) + (bodyScore * 0.1);
        return {
            score: boundPose ? ((templateScore * 0.65) + (defaultScore * 0.35)) : defaultScore,
            sample,
            debug: {
                primitiveScore,
                fingerScore,
                thumbScore,
                bodyScore,
                templateScore,
                usingBoundPose: !!boundPose
            }
        };
    }).sort((left, right) => right.score - left.score);

    return {
        score: candidates[0].score,
        sample: candidates[0].sample,
        debug: candidates[0].debug,
        handsVisible: hands.length
    };
}

function drawFivePointTracking(results) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    const pose = results.poseLandmarks || [];
    [pose[0], pose[11], pose[12]].filter(Boolean).forEach(point => {
        canvasCtx.beginPath();
        canvasCtx.fillStyle = "rgba(251, 113, 133, 0.9)";
        canvasCtx.arc(point.x * outputCanvas.width, point.y * outputCanvas.height, 5, 0, Math.PI * 2);
        canvasCtx.fill();
    });

    [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).forEach(hand => {
        const sparse = extractSparseHand(hand);
        if (!sparse) {
            return;
        }
        sparse.forEach((point, index) => {
            canvasCtx.beginPath();
            canvasCtx.fillStyle = index === 0 ? "rgba(251, 191, 36, 0.95)" : "rgba(56, 189, 248, 0.95)";
            canvasCtx.arc(point.x * outputCanvas.width, point.y * outputCanvas.height, index === 0 ? 7 : 6, 0, Math.PI * 2);
            canvasCtx.fill();
        });
    });

    canvasCtx.restore();
}

function renderLetterButtons() {
    LETTER_BUTTONS.forEach(button => {
        const isActive = button.dataset.letter === currentLetter;
        button.classList.toggle("is-active", isActive);
        button.setAttribute("aria-pressed", String(isActive));
    });
}

function renderStaticCopy() {
    const spec = LETTER_SPECS[currentLetter];
    letterTitle.textContent = spec.title;
    targetCopy.textContent = spec.description;
    primitiveLabel.textContent = `Primitive: ${spec.primitiveHint}`;
}

function renderBreakdown(debug) {
    scoreBreakdown.innerHTML = "";
    const items = !debug
        ? [{ label: "Waiting", value: "No score yet.", badge: "Idle" }]
        : [
            { label: "Primitive", value: `${Math.round(debug.primitiveScore * 100)}%`, badge: LETTER_SPECS[currentLetter].primitiveHint },
            { label: "Finger rule", value: `${Math.round(debug.fingerScore * 100)}%`, badge: "Fingers" },
            { label: "Thumb rule", value: `${Math.round(debug.thumbScore * 100)}%`, badge: "Thumb" },
            { label: "Bound pose", value: debug.usingBoundPose ? `${Math.round(debug.templateScore * 100)}%` : "Not used", badge: debug.usingBoundPose ? "Saved" : "Default" }
        ];

    items.forEach(item => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        row.innerHTML = `<span>${item.label}</span><span class="gesture-check-badge is-neutral">${item.value}${item.badge ? ` - ${item.badge}` : ""}</span>`;
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
        { label: "Letter", value: LETTER_SPECS[currentLetter].title, badge: currentLetter, tone: "is-good" },
        { label: "Matcher", value: `Primitive-first recognition using ${LETTER_SPECS[currentLetter].primitiveHint}.`, badge: "Handshape", tone: "is-good" },
        { label: "Camera state", value: cameraReady ? "Live frames are being scored." : "Camera stream is not ready yet.", badge: cameraReady ? "Live" : "Waiting", tone: cameraReady ? "is-good" : "is-neutral" },
        { label: "Permission", value: `Camera permission state: ${permissionState}.`, badge: permissionState, tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral" },
        { label: "Tracked hands", value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible.", badge: `${trackedHands}`, tone: trackedHands ? "is-good" : "is-neutral" }
    ];

    if (latestDebug) {
        rows.push({
            label: "Debug",
            value: `Primitive ${Math.round(latestDebug.primitiveScore * 100)}%, fingers ${Math.round(latestDebug.fingerScore * 100)}%, thumb ${Math.round(latestDebug.thumbScore * 100)}%, bound pose ${Math.round((latestDebug.templateScore || 0) * 100)}%.`,
            badge: "Live",
            tone: "is-good"
        });
        rows.push({
            label: "Bound pose status",
            value: latestDebug.usingBoundPose ? `A saved pose is active for ${currentLetter}.` : `No saved pose for ${currentLetter}.`,
            badge: latestDebug.usingBoundPose ? "Active" : "Default",
            tone: latestDebug.usingBoundPose ? "is-good" : "is-neutral"
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
        row.innerHTML = `<div class="gesture-diagnostic-copy"><div class="gesture-diagnostic-label">${item.label}</div><div class="gesture-diagnostic-value">${item.value}</div></div><span class="gesture-check-badge ${item.tone || "is-neutral"}">${item.badge}</span>`;
        diagnosticsList.appendChild(row);
    });
}

async function collectDiagnostics(summary = "") {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = summary || `This page tests ${currentLetter} through a base handshape detector plus letter-specific rules.`;
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
    latestScoredSample = result.sample;
    trackedHands = result.handsVisible;
    renderBreakdown(result.debug);

    let holdProgress = 0;
    let statusText = `Show ${currentLetter} to the camera.`;
    if (!result.handsVisible) {
        holdStartedAt = 0;
        statusText = "Show one clear hand to the camera.";
    } else if (result.handsVisible > 1) {
        holdStartedAt = 0;
        statusText = "Use one main hand for this alphabet test.";
    } else if (result.score >= SCORE_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        statusText = elapsed >= HOLD_SECONDS
            ? `${currentLetter} matched.`
            : `Good ${currentLetter} pose. Keep holding.`;
    } else {
        holdStartedAt = 0;
        statusText = `Adjust the handshape until it looks more like ${currentLetter}.`;
    }

    renderStatus(result.score, holdProgress, statusText);
    collectDiagnostics().catch(console.error);
}

function bindCurrentPose() {
    if (!latestScoredSample?.flattenedSparse) {
        renderStatus(0, 0, `Show ${currentLetter} first, then bind the current pose.`);
        return;
    }
    saveBoundPose(currentLetter, latestScoredSample);
    renderStatus(Math.max(Number(gestureScore.textContent.replace("%", "")) / 100 || 0, 0), 0, `Current ${currentLetter} pose saved.`);
    collectDiagnostics(`Bound pose saved for ${currentLetter}.`).catch(console.error);
}

function resetBoundPose() {
    clearBoundPose(currentLetter);
    renderStatus(0, 0, `Saved ${currentLetter} pose cleared.`);
    collectDiagnostics(`Bound pose cleared for ${currentLetter}.`).catch(console.error);
}

function setLetter(letter) {
    currentLetter = letter;
    boundPose = loadBoundPose(letter);
    latestDebug = null;
    latestScoredSample = null;
    holdStartedAt = 0;
    renderLetterButtons();
    renderStaticCopy();
    renderBreakdown(null);
    renderStatus(0, 0, `Show ${currentLetter} to the camera.`);
    collectDiagnostics(`Switched to ${currentLetter}.`).catch(console.error);
}

function onHolisticResults(results) {
    drawFivePointTracking(results);
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    handleScoredFrame(scoreCurrentLetter(results));
}

async function startCamera() {
    stopLoop();
    lastCameraError = "";
    cameraState.textContent = "Starting camera...";
    renderStatus(0, 0, `Waiting for camera access for ${currentLetter}.`);
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

LETTER_BUTTONS.forEach(button => {
    button.addEventListener("click", () => setLetter(button.dataset.letter));
});
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
bindPoseBtn.addEventListener("click", bindCurrentPose);
resetPoseBtn.addEventListener("click", resetBoundPose);
window.addEventListener("beforeunload", () => stopLoop());

renderLetterButtons();
renderStaticCopy();
renderBreakdown(null);
collectDiagnostics().then(() => startCamera()).catch(console.error);
