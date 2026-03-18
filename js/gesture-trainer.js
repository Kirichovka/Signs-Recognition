import {
    describeCameraError,
    drawHolisticResults,
    getCameraPermissionState,
    prettifyLabel,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-9";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.72;
const LETTER_A_THRESHOLD = 0.4;
const FRAME_SKIP = 4;
const ALPHABET_TOLERANCE = 0.18;
const WORD_TOLERANCE = 0.24;
const SPARSE_POINTS = [0, 4, 8, 12, 16, 20];

const WORD_LABELS = ["HELLO", "BYE", "YES", "NO", "PLEASE", "SORRY", "HELP", "THANKYOU", "WELCOME1", "EAT1", "DRINK1", "WATER", "MOTHER", "FATHER", "FAMILY", "HOME", "HOUSE", "SCHOOL", "WORK", "FRIEND", "LOVE", "WANT1", "NEED", "COME", "COMEHERE", "GO", "STOP", "FINISH", "GOOD", "BAD", "HAPPY", "SAD", "NOW", "MORE", "NOT", "KNOW", "DONTKNOW", "NOTUNDERSTAND", "GOAHEAD", "GREAT"];
const ALPHABET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"];

const HAND_TEMPLATES = {
    fist: [[0, 0], [-0.8, -0.15], [-0.35, -0.42], [0, -0.38], [0.28, -0.32], [0.55, -0.25]],
    flat: [[0, 0], [-0.95, -0.05], [-0.35, -1.05], [0, -1.08], [0.38, -1.02], [0.72, -0.95]],
    open: [[0, 0], [-0.95, -0.22], [-0.45, -1.12], [0, -1.25], [0.42, -1.1], [0.82, -0.92]],
    curve: [[0, 0], [-0.7, -0.4], [-0.42, -0.95], [0, -1.12], [0.45, -0.96], [0.72, -0.55]],
    point: [[0, 0], [-0.75, -0.18], [-0.05, -1.08], [0.1, -0.35], [0.3, -0.25], [0.52, -0.2]],
    pinch: [[0, 0], [-0.08, -0.58], [-0.02, -0.6], [0.14, -0.52], [0.34, -0.34], [0.58, -0.22]],
    ok: [[0, 0], [-0.06, -0.58], [-0.02, -0.6], [0.08, -1.02], [0.4, -0.98], [0.76, -0.9]],
    gun: [[0, 0], [-1.0, -0.1], [0.65, -0.12], [0.2, -0.28], [0.34, -0.22], [0.52, -0.16]],
    two: [[0, 0], [-0.78, -0.15], [-0.22, -1.05], [0.22, -1.08], [0.26, -0.32], [0.48, -0.22]],
    lshape: [[0, 0], [-1.02, -0.05], [-0.18, -1.08], [0.12, -0.28], [0.32, -0.18], [0.56, -0.08]],
    three: [[0, 0], [-0.82, -0.16], [-0.38, -1.05], [0.0, -1.12], [0.42, -1.0], [0.52, -0.22]],
    pinky: [[0, 0], [-0.78, -0.12], [-0.14, -0.22], [0.08, -0.16], [0.28, -0.12], [0.68, -1.02]],
    yshape: [[0, 0], [-1.02, -0.22], [-0.18, -0.18], [0.04, -0.14], [0.26, -0.1], [0.92, -0.98]],
    hook: [[0, 0], [-0.8, -0.12], [-0.04, -0.82], [0.1, -0.26], [0.3, -0.2], [0.54, -0.16]],
    vshape: [[0, 0], [-0.82, -0.14], [-0.34, -1.06], [0.26, -1.1], [0.24, -0.32], [0.48, -0.22]]
};

const MODE_CONFIG = {
    words: {
        title: "Everyday Words",
        summary: "Practice common ASL words with template matching on the last tracked frame.",
        targetLabel: "Target Sign",
        tips: "Match the hand overlay, keep the body centered, and freeze the final pose for about one second.",
        tolerance: WORD_TOLERANCE,
        labels: WORD_LABELS
    },
    alphabet: {
        title: "Alphabet",
        summary: "Practice single ASL letters with handshape templates and a per-point tolerance on the last frame.",
        targetLabel: "Target Letter",
        tips: "Use one clear hand, center it in the camera, and hold the handshape still until the points line up.",
        tolerance: ALPHABET_TOLERANCE,
        labels: ALPHABET_LABELS
    }
};

const ALPHABET_SPEC = {
    A: { shape: "fist", zone: "neutral", hands: 1, hint: "Closed fist with the thumb resting along the side." },
    B: { shape: "flat", zone: "neutral", hands: 1, hint: "Flat hand with fingers together and thumb folded across the palm." },
    C: { shape: "curve", zone: "neutral", hands: 1, hint: "Curve the hand into a clear C-shape." },
    D: { shape: "point", zone: "neutral", hands: 1, hint: "Index finger up, thumb near the middle finger." },
    E: { shape: "fist", zone: "neutral", hands: 1, hint: "Curl all fingers down toward the thumb." },
    F: { shape: "ok", zone: "neutral", hands: 1, hint: "Index finger and thumb make a small circle." },
    G: { shape: "gun", zone: "neutral", hands: 1, hint: "Index finger and thumb extend sideways with a narrow gap." },
    H: { shape: "two", zone: "neutral", hands: 1, hint: "Index and middle fingers extend together." },
    I: { shape: "pinky", zone: "neutral", hands: 1, hint: "Only the pinky is raised." },
    K: { shape: "vshape", zone: "neutral", hands: 1, hint: "Make a V-shape and place the thumb at the base." },
    L: { shape: "lshape", zone: "neutral", hands: 1, hint: "Index finger up and thumb out to form an L." },
    M: { shape: "fist", zone: "neutral", hands: 1, hint: "Thumb tucked under the first three fingers." },
    N: { shape: "fist", zone: "neutral", hands: 1, hint: "Thumb tucked under the first two fingers." },
    O: { shape: "curve", zone: "neutral", hands: 1, hint: "Curve all fingertips toward the thumb to make an O." },
    P: { shape: "vshape", zone: "neutral", hands: 1, hint: "Like K, but hold the hand a bit lower." },
    Q: { shape: "gun", zone: "neutral", hands: 1, hint: "Like G, but keep the hand slightly lower." },
    R: { shape: "two", zone: "neutral", hands: 1, hint: "Lift the index and middle fingers and keep them close like a crossed pair." },
    S: { shape: "fist", zone: "neutral", hands: 1, hint: "Closed fist with the thumb across the front." },
    T: { shape: "fist", zone: "neutral", hands: 1, hint: "Thumb tucked between the index and middle fingers." },
    U: { shape: "two", zone: "neutral", hands: 1, hint: "Index and middle fingers point up together." },
    V: { shape: "vshape", zone: "neutral", hands: 1, hint: "Index and middle fingers point up and spread apart." },
    W: { shape: "three", zone: "neutral", hands: 1, hint: "Three fingers point up." },
    X: { shape: "hook", zone: "neutral", hands: 1, hint: "Raise the index finger in a hooked shape." },
    Y: { shape: "yshape", zone: "neutral", hands: 1, hint: "Thumb and pinky extended, middle fingers folded." }
};

const WORD_SPEC = {
    HELLO: { shape: "flat", zone: "head", hands: 1, hint: "Flat greeting hand near the face." },
    BYE: { shape: "flat", zone: "head", hands: 1, hint: "Flat hand near the face in a goodbye pose." },
    YES: { shape: "fist", zone: "neutral", hands: 1, hint: "Compact fist held in neutral signing space." },
    NO: { shape: "pinch", zone: "mouth", hands: 1, hint: "Pinched fingers held near the mouth." },
    PLEASE: { shape: "flat", zone: "chest", hands: 1, hint: "Flat hand on the chest area." },
    SORRY: { shape: "fist", zone: "chest", hands: 1, hint: "Compact handshape centered on the chest." },
    HELP: { shape: "lshape", zone: "chest", hands: 2, hint: "One helping hand over another at chest level." },
    THANKYOU: { shape: "flat", zone: "mouth", hands: 1, hint: "Flat hand starting near the mouth." },
    WELCOME1: { shape: "open", zone: "chest", hands: 2, hint: "Open hands in front of the chest." },
    EAT1: { shape: "pinch", zone: "mouth", hands: 1, hint: "Pinched fingertips near the mouth." },
    DRINK1: { shape: "curve", zone: "mouth", hands: 1, hint: "Curved hand like holding a cup near the mouth." },
    WATER: { shape: "two", zone: "mouth", hands: 1, hint: "Two-finger handshape near the mouth." },
    MOTHER: { shape: "open", zone: "head", hands: 1, hint: "Open hand near the upper face." },
    FATHER: { shape: "open", zone: "head", hands: 1, hint: "Open hand near the forehead." },
    FAMILY: { shape: "open", zone: "chest", hands: 2, hint: "Two open hands held in front of the chest." },
    HOME: { shape: "flat", zone: "neutral", hands: 2, hint: "Two hands brought together in front of the face or chest." },
    HOUSE: { shape: "flat", zone: "chest", hands: 2, hint: "Two hands shape a house outline." },
    SCHOOL: { shape: "flat", zone: "chest", hands: 2, hint: "Two flat hands meet at chest height." },
    WORK: { shape: "fist", zone: "chest", hands: 2, hint: "Two compact hands aligned in front of the torso." },
    FRIEND: { shape: "hook", zone: "chest", hands: 2, hint: "Hooked handshapes linked at chest level." },
    LOVE: { shape: "fist", zone: "chest", hands: 2, hint: "Closed arms or hands in front of the chest." },
    WANT1: { shape: "open", zone: "chest", hands: 1, hint: "Open handshape in front of the chest." },
    NEED: { shape: "open", zone: "chest", hands: 1, hint: "Open handshape held strongly in front of the chest." },
    COME: { shape: "open", zone: "neutral", hands: 1, hint: "Open hand in neutral space." },
    COMEHERE: { shape: "open", zone: "neutral", hands: 1, hint: "Open hand in neutral space, ready to call inward." },
    GO: { shape: "point", zone: "neutral", hands: 1, hint: "Forward pointing handshape in neutral space." },
    STOP: { shape: "flat", zone: "chest", hands: 2, hint: "Flat hands held in a stopping configuration." },
    FINISH: { shape: "open", zone: "chest", hands: 2, hint: "Open hands in front of the chest." },
    GOOD: { shape: "flat", zone: "mouth", hands: 1, hint: "Flat hand near the mouth." },
    BAD: { shape: "flat", zone: "mouth", hands: 1, hint: "Flat hand near the mouth with a downward finish." },
    HAPPY: { shape: "open", zone: "chest", hands: 2, hint: "Open hands or chest-level brushing pose." },
    SAD: { shape: "open", zone: "head", hands: 2, hint: "Open hands higher on the face." },
    NOW: { shape: "yshape", zone: "neutral", hands: 1, hint: "Compact Y-like handshape in neutral space." },
    MORE: { shape: "pinch", zone: "chest", hands: 2, hint: "Pinched handshapes meeting at chest level." },
    NOT: { shape: "two", zone: "neutral", hands: 1, hint: "Two-finger handshape in neutral space." },
    KNOW: { shape: "flat", zone: "head", hands: 1, hint: "Flat handshape near the forehead." },
    DONTKNOW: { shape: "flat", zone: "head", hands: 1, hint: "Flat handshape near the head." },
    NOTUNDERSTAND: { shape: "flat", zone: "head", hands: 1, hint: "Flat handshape near the head." },
    GOAHEAD: { shape: "flat", zone: "neutral", hands: 1, hint: "Flat hand released in neutral space." },
    GREAT: { shape: "open", zone: "chest", hands: 2, hint: "Open, strong handshape centered on the chest." }
};

const modeWordsBtn = document.getElementById("mode-words-btn");
const modeAlphabetBtn = document.getElementById("mode-alphabet-btn");
const trainerSubtitle = document.getElementById("trainer-subtitle");
const gestureTargetLabel = document.getElementById("gesture-target-label");
const gestureTitle = document.getElementById("gesture-title");
const gestureInstruction = document.getElementById("gesture-instruction");
const gestureEmoji = document.getElementById("gesture-emoji");
const gestureStep = document.getElementById("gesture-step");
const holdProgressBar = document.getElementById("hold-progress-bar");
const holdProgressValue = document.getElementById("hold-progress-value");
const gestureScore = document.getElementById("gesture-score");
const gestureStatus = document.getElementById("gesture-status");
const fingerChecklist = document.getElementById("finger-checklist");
const gestureGuideTitle = document.getElementById("gesture-guide-title");
const gestureGuideBadge = document.getElementById("gesture-guide-badge");
const gestureGuideDescription = document.getElementById("gesture-guide-description");
const gestureGuideHand = document.getElementById("gesture-guide-hand");
const gestureGuideArrow = document.getElementById("gesture-guide-arrow");
const gestureGuideChips = document.getElementById("gesture-guide-chips");
const gestureGuideSteps = document.getElementById("gesture-guide-steps");
const cameraState = document.getElementById("camera-state");
const cameraHelpText = document.getElementById("camera-help-text");
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

let currentMode = "words";
let gestures = [];
let currentGestureIndex = 0;
let holdStartedAt = 0;
let lastSuccess = false;
let activeStream = null;
let animationFrameId = 0;
let holistic = null;
let frameCounter = 0;
let lastCameraError = "";
let cameraReady = false;
let trackedHands = 0;
let latestFrameAnalysis = null;
let latestPredictions = [];
let latestGeometryDebug = null;

function getModeConfig() {
    return MODE_CONFIG[currentMode];
}

function isAlphabetMode() {
    return currentMode === "alphabet";
}

function getGestureSpec(label) {
    return isAlphabetMode() ? ALPHABET_SPEC[label] : WORD_SPEC[label];
}

function buildGestures(labels) {
    return labels.map(label => ({
        id: label,
        title: prettifyLabel(label),
        instruction: isAlphabetMode()
            ? `Shape the hand for "${prettifyLabel(label)}" and line up the points with the target overlay.`
            : `Show the sign for "${prettifyLabel(label)}" and match the target pose on the camera overlay.`
    }));
}

function getCurrentGesture() {
    return gestures[currentGestureIndex] || gestures[0];
}

function getGuide(gesture) {
    const spec = getGestureSpec(gesture.id);
    const zoneTitles = {
        head: "Head or face level",
        mouth: "Near the mouth",
        chest: "Upper chest space",
        neutral: "Neutral signing space"
    };
    return {
        title: isAlphabetMode() ? `Letter ${gesture.title}` : zoneTitles[spec.zone] || "Neutral signing space",
        description: spec.hint,
        chips: [
            `${spec.hands} hand${spec.hands > 1 ? "s" : ""}`,
            isAlphabetMode() ? "Last-frame match" : zoneTitles[spec.zone] || "Neutral signing space",
            `Tolerance ${getModeConfig().tolerance.toFixed(2)}`
        ],
        steps: [
            spec.hint,
            spec.hands > 1 ? "Keep both hands visible and balanced in the frame." : "Keep one clear dominant hand visible.",
            "Move until your fingertips sit close to the target dots, then hold the final pose."
        ],
        visual: {
            x: spec.zone === "head" ? "50%" : spec.zone === "mouth" ? "50%" : "50%",
            y: spec.zone === "head" ? "28%" : spec.zone === "mouth" ? "38%" : spec.zone === "chest" ? "54%" : "52%",
            rotation: 0,
            scale: 1,
            movementTone: isAlphabetMode() ? "Last-frame template" : "Pose template",
            isStill: true
        }
    };
}

function renderModeButtons() {
    modeWordsBtn.classList.toggle("is-active", currentMode === "words");
    modeAlphabetBtn.classList.toggle("is-active", currentMode === "alphabet");
    modeWordsBtn.setAttribute("aria-pressed", String(currentMode === "words"));
    modeAlphabetBtn.setAttribute("aria-pressed", String(currentMode === "alphabet"));
}

function renderGuideCard() {
    const guide = getGuide(getCurrentGesture());
    gestureGuideTitle.textContent = guide.title;
    gestureGuideBadge.textContent = guide.visual.movementTone;
    gestureGuideDescription.textContent = guide.description;
    gestureGuideHand.style.setProperty("--guide-x", guide.visual.x);
    gestureGuideHand.style.setProperty("--guide-y", guide.visual.y);
    gestureGuideArrow.classList.add("is-still");
    gestureGuideChips.innerHTML = "";
    gestureGuideSteps.innerHTML = "";
    guide.chips.forEach(text => {
        const chip = document.createElement("span");
        chip.className = "gesture-guide-chip";
        chip.textContent = text;
        gestureGuideChips.appendChild(chip);
    });
    guide.steps.forEach((text, index) => {
        const row = document.createElement("div");
        row.className = "gesture-guide-step";
        row.innerHTML = `<div class="gesture-guide-step-index">${index + 1}</div><div class="gesture-guide-step-copy">${text}</div>`;
        gestureGuideSteps.appendChild(row);
    });
}

function renderGesturePreview(gesture) {
    if (isAlphabetMode()) {
        gestureEmoji.innerHTML = `<svg viewBox="0 0 100 100" class="gesture-preview-svg" aria-hidden="true"><defs><linearGradient id="alphabetPreviewBg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#2563eb" /><stop offset="100%" stop-color="#38bdf8" /></linearGradient></defs><rect x="2" y="2" width="96" height="96" rx="24" fill="url(#alphabetPreviewBg)" /><text x="50" y="58" text-anchor="middle" font-size="36" font-weight="800" fill="rgba(255,255,255,0.98)" font-family="Arial, sans-serif">${gesture.title}</text><text x="50" y="82" text-anchor="middle" font-size="10" font-weight="700" fill="rgba(255,255,255,0.78)" font-family="Arial, sans-serif">POINT MATCH</text></svg>`;
        return;
    }
    gestureEmoji.innerHTML = `<svg viewBox="0 0 100 100" class="gesture-preview-svg" aria-hidden="true"><defs><linearGradient id="gesturePreviewBg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#f97316" /><stop offset="100%" stop-color="#fb923c" /></linearGradient></defs><rect x="2" y="2" width="96" height="96" rx="24" fill="url(#gesturePreviewBg)" /><circle cx="50" cy="24" r="12" fill="rgba(255,255,255,0.24)" /><rect x="27" y="38" width="46" height="26" rx="12" fill="rgba(255,255,255,0.18)" /><text x="50" y="82" text-anchor="middle" font-size="10" font-weight="700" fill="rgba(255,255,255,0.78)" font-family="Arial, sans-serif">LAST FRAME</text></svg>`;
}

function renderGestureCard() {
    const gesture = getCurrentGesture();
    const mode = getModeConfig();
    trainerSubtitle.textContent = mode.summary;
    gestureTargetLabel.textContent = mode.targetLabel;
    gestureTitle.textContent = gesture.title;
    gestureInstruction.textContent = gesture.instruction;
    cameraHelpText.textContent = mode.tips;
    gestureStep.textContent = `${currentGestureIndex + 1} / ${gestures.length}`;
    renderGesturePreview(gesture);
    renderGuideCard();
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
        fingerChecklist.innerHTML = "<div class=\"gesture-check-row\"><span>Waiting</span><span class=\"gesture-check-badge is-neutral\">No match yet</span></div>";
        return;
    }
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        row.innerHTML = `<span>${index + 1}. ${prettifyLabel(item.label)}</span><span class="gesture-check-badge ${index === 0 ? "is-good" : "is-neutral"}">${Math.round(item.score * 100)}%</span>`;
        fingerChecklist.appendChild(row);
    });
}

function resetTrackingState() {
    holdStartedAt = 0;
    lastSuccess = false;
    trackedHands = 0;
    latestFrameAnalysis = null;
    latestPredictions = [];
}

function setGesture(index) {
    currentGestureIndex = (index + gestures.length) % gestures.length;
    resetTrackingState();
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, "Move your hand until the live landmarks line up with the target points.");
}

function averagePoint(points) {
    if (!points.length) {
        return null;
    }
    return {
        x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
        y: points.reduce((sum, point) => sum + point.y, 0) / points.length
    };
}

function visiblePoint(point, minVisibility = 0.2) {
    return !!point && (point.visibility === undefined || point.visibility >= minVisibility);
}

function getZoneCenter(results, zone) {
    const pose = results.poseLandmarks || [];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const nose = pose[0];
    const mouthLeft = pose[9];
    const mouthRight = pose[10];
    const shouldersVisible = visiblePoint(leftShoulder) && visiblePoint(rightShoulder);
    const shoulderCenter = shouldersVisible ? averagePoint([leftShoulder, rightShoulder]) : { x: 0.5, y: 0.5 };
    const shoulderSpan = shouldersVisible ? Math.max(0.08, Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y)) : 0.18;
    const mouthCenter = visiblePoint(mouthLeft) && visiblePoint(mouthRight)
        ? averagePoint([mouthLeft, mouthRight])
        : visiblePoint(nose)
            ? { x: nose.x, y: nose.y + shoulderSpan * 0.1 }
            : { x: shoulderCenter.x, y: shoulderCenter.y - shoulderSpan * 0.45 };
    if (zone === "head") { return { x: shoulderCenter.x, y: shoulderCenter.y - shoulderSpan * 0.7 }; }
    if (zone === "mouth") { return mouthCenter; }
    if (zone === "chest") { return { x: shoulderCenter.x, y: shoulderCenter.y + shoulderSpan * 0.22 }; }
    return { x: shoulderCenter.x, y: shoulderCenter.y + shoulderSpan * 0.32 };
}

function extractSparseHand(handLandmarks) {
    if (!handLandmarks?.length) { return null; }
    return SPARSE_POINTS.map(index => handLandmarks[index] || handLandmarks[0]);
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

const TEMPLATE_VECTORS = Object.fromEntries(Object.entries(HAND_TEMPLATES).map(([shape, points]) => [shape, pairwiseVector(points.map(([x, y]) => ({ x, y })))]));

function compareVectors(left, right, tolerance) {
    const diff = left.reduce((sum, value, index) => sum + Math.abs(value - right[index]), 0) / left.length;
    return Math.max(0, 1 - diff / tolerance);
}

function scoreHandShape(handLandmarks, shapeName) {
    const sparse = extractSparseHand(handLandmarks);
    if (!sparse) { return 0; }
    return compareVectors(pairwiseVector(sparse), TEMPLATE_VECTORS[shapeName], getModeConfig().tolerance);
}

function zoneScore(results, zone, handLandmarks) {
    if (isAlphabetMode() || !handLandmarks?.length) { return handLandmarks?.length ? 1 : 0; }
    const handCenter = averagePoint(handLandmarks);
    const targetCenter = getZoneCenter(results, zone);
    const pose = results.poseLandmarks || [];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const shoulderSpan = visiblePoint(leftShoulder) && visiblePoint(rightShoulder)
        ? Math.max(0.08, Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y))
        : 0.18;
    const distance = Math.hypot(handCenter.x - targetCenter.x, handCenter.y - targetCenter.y);
    return Math.max(0, 1 - distance / (shoulderSpan * 0.9));
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function distance2D(left, right) {
    return Math.hypot(left.x - right.x, left.y - right.y);
}

function average(values) {
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function getBodyFrame(results) {
    const pose = results.poseLandmarks || [];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const shouldersVisible = visiblePoint(leftShoulder) && visiblePoint(rightShoulder);
    const center = shouldersVisible ? averagePoint([leftShoulder, rightShoulder]) : { x: 0.5, y: 0.52 };
    const scale = shouldersVisible
        ? Math.max(0.08, distance2D(leftShoulder, rightShoulder))
        : 0.18;
    return { center, scale };
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

function scoreLetterAForHand(results, handLandmarks) {
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

    const fingerCurl = [
        scoreCurledFinger(indexMcp, indexPip, indexDip, indexTip),
        scoreCurledFinger(middleMcp, middlePip, middleDip, middleTip),
        scoreCurledFinger(ringMcp, ringPip, ringDip, ringTip),
        scoreCurledFinger(pinkyMcp, pinkyPip, pinkyDip, pinkyTip)
    ];
    const curledScore = average(fingerCurl.map(item => item.score));

    const thumbHorizontal = Math.abs((thumbTip.x - indexMcp.x) / handScale);
    const thumbVertical = Math.abs((thumbTip.y - indexMcp.y) / handScale);
    const thumbScore = (scoreProximity(thumbHorizontal, 0.58, 0.42) * 0.7) + (scoreProximity(thumbVertical, 0.12, 0.28) * 0.3);

    const bodyFrame = getBodyFrame(results);
    const wristX = (wrist.x - bodyFrame.center.x) / bodyFrame.scale;
    const wristY = (wrist.y - bodyFrame.center.y) / bodyFrame.scale;
    const bodyPositionScore = (scoreProximity(Math.abs(wristX), 0.55, 0.8) * 0.45) + (scoreProximity(wristY, 0.15, 0.7) * 0.55);

    const finalScore = (curledScore * 0.45) + (thumbScore * 0.3) + (bodyPositionScore * 0.25);
    return {
        score: finalScore,
        debug: {
            curledScore,
            thumbScore,
            bodyPositionScore,
            wristX,
            wristY,
            fingerCurl,
            thumbHorizontal,
            thumbVertical
        }
    };
}

function scoreLetterA(results) {
    const hands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean);
    if (!hands.length) {
        return { label: "A", score: 0, debug: null };
    }
    const candidates = hands.map(hand => scoreLetterAForHand(results, hand)).sort((left, right) => right.score - left.score);
    return {
        label: "A",
        score: candidates[0].score,
        debug: candidates[0].debug
    };
}

function bestHandScores(results, spec) {
    const hands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean);
    if (!hands.length) { return { shape: 0, zone: 0, visibleHands: 0 }; }
    const scored = hands.map(hand => ({ hand, shape: scoreHandShape(hand, spec.shape), zone: zoneScore(results, spec.zone, hand) }));
    scored.sort((a, b) => (b.shape + b.zone) - (a.shape + a.zone));
    if (spec.hands === 1) { return { shape: scored[0].shape, zone: scored[0].zone, visibleHands: hands.length }; }
    const top = scored.slice(0, 2);
    return {
        shape: top.reduce((sum, item) => sum + item.shape, 0) / top.length,
        zone: top.reduce((sum, item) => sum + item.zone, 0) / top.length,
        visibleHands: hands.length
    };
}

function scoreGesture(results, label) {
    if (isAlphabetMode() && label === "A") {
        const aScore = scoreLetterA(results);
        return { label, score: aScore.score, debug: aScore.debug };
    }
    const spec = getGestureSpec(label);
    const handInfo = bestHandScores(results, spec);
    const handCountScore = Math.min(1, handInfo.visibleHands / spec.hands);
    const zone = isAlphabetMode() ? 1 : handInfo.zone;
    const score = isAlphabetMode()
        ? (handInfo.shape * 0.85) + (handCountScore * 0.15)
        : (handInfo.shape * 0.55) + (zone * 0.3) + (handCountScore * 0.15);
    return { label, score };
}

function buildPredictions(results) {
    const scored = gestures.map(gesture => scoreGesture(results, gesture.id)).sort((a, b) => b.score - a.score);
    latestGeometryDebug = scored.find(item => item.label === "A")?.debug || null;
    return scored.slice(0, 5);
}

function analyzeCurrentFrame(results) {
    const handsVisible = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    const targetSpec = getGestureSpec(getCurrentGesture().id);
    const warnings = [];
    if (handsVisible < targetSpec.hands) {
        warnings.push(targetSpec.hands === 2 ? "This target needs both hands visible." : "Show one clear dominant hand.");
    }
    if (isAlphabetMode() && handsVisible > 1) {
        warnings.push("Use one main hand for alphabet mode.");
    }
    return {
        handsVisible,
        zoneTarget: targetSpec.zone,
        zoneOk: warnings.length === 0,
        primaryWarning: warnings[0] || ""
    };
}

function drawTemplateOverlay(results, gesture) {
    const spec = getGestureSpec(gesture.id);
    const center = getZoneCenter(results, spec.zone);
    const width = outputCanvas.width || 1280;
    const height = outputCanvas.height || 720;
    const scale = Math.min(width, height) * (isAlphabetMode() ? 0.1 : 0.085);
    const dotPoints = HAND_TEMPLATES[spec.shape].map(([x, y]) => ({
        x: (center.x * width) + (x * scale),
        y: (center.y * height) + (y * scale)
    }));

    canvasCtx.save();
    canvasCtx.strokeStyle = "rgba(255,255,255,0.9)";
    canvasCtx.lineWidth = 2;
    canvasCtx.setLineDash([4, 5]);
    canvasCtx.beginPath();
    dotPoints.forEach((point, index) => {
        if (index === 0) {
            canvasCtx.moveTo(point.x, point.y);
        } else {
            canvasCtx.lineTo(point.x, point.y);
        }
    });
    canvasCtx.stroke();
    canvasCtx.setLineDash([]);
    dotPoints.forEach((point, index) => {
        canvasCtx.beginPath();
        canvasCtx.fillStyle = index === 0 ? "rgba(251,191,36,0.95)" : "rgba(56,189,248,0.95)";
        canvasCtx.arc(point.x, point.y, index === 0 ? 6 : 5, 0, Math.PI * 2);
        canvasCtx.fill();
    });
    canvasCtx.restore();
}

function evaluatePredictions(predictions) {
    latestPredictions = predictions;
    const target = getCurrentGesture();
    const targetPrediction = predictions.find(item => item.label === target.id) || null;
    const best = predictions[0] || null;
    const targetScore = targetPrediction?.score || 0;

    let holdProgress = 0;
    let stableSuccess = false;
    let statusText = "Move your hand until the live landmarks line up with the target points.";

    if (latestFrameAnalysis?.primaryWarning) {
        holdStartedAt = 0;
        statusText = latestFrameAnalysis.primaryWarning;
    } else if (targetScore >= (isAlphabetMode() && target.id === "A" ? LETTER_A_THRESHOLD : SCORE_THRESHOLD)) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        stableSuccess = elapsed >= HOLD_SECONDS;
        statusText = stableSuccess
            ? `Matched ${target.title}. Moving to the next ${isAlphabetMode() ? "letter" : "sign"}.`
            : `Good alignment for ${target.title}. Keep holding the pose.`;
    } else if (best) {
        holdStartedAt = 0;
        statusText = best.score >= 0.45
            ? `Closest match is ${prettifyLabel(best.label)} at ${Math.round(best.score * 100)}%. Adjust toward ${target.title}.`
            : `Not aligned yet. Bring your landmarks closer to the target dots for ${target.title}.`;
    }

    renderStatus(targetScore, holdProgress, statusText);
    if (stableSuccess && !lastSuccess) {
        lastSuccess = true;
        window.setTimeout(() => setGesture(currentGestureIndex + 1), 450);
        return;
    }
    lastSuccess = stableSuccess;
}

function renderDiagnosticRows(rows) {
    diagnosticsList.innerHTML = "";
    rows.forEach(rowData => {
        const row = document.createElement("div");
        row.className = "gesture-diagnostic-row";
        row.innerHTML = `<div class="gesture-diagnostic-copy"><div class="gesture-diagnostic-label">${rowData.label}</div><div class="gesture-diagnostic-value">${rowData.value}</div></div><span class="gesture-check-badge ${rowData.tone || "is-neutral"}">${rowData.badge}</span>`;
        diagnosticsList.appendChild(row);
    });
}

async function collectDiagnostics(extra = {}) {
    const permissionState = await getCameraPermissionState();
    const rows = [
        { label: "Mode", value: getModeConfig().title, badge: getModeConfig().title, tone: "is-good" },
        { label: "Matcher", value: `Template score with tolerance ${getModeConfig().tolerance.toFixed(2)}.`, badge: "Template", tone: "is-good" },
        { label: isAlphabetMode() ? "Target letter" : "Target sign", value: getCurrentGesture().title, badge: `${currentGestureIndex + 1}/${gestures.length}`, tone: "is-good" },
        { label: "Camera state", value: cameraReady ? "Video frames are available for landmark tracking." : "Camera stream is not ready yet.", badge: cameraReady ? "Live" : "Waiting", tone: cameraReady ? "is-good" : "is-neutral" },
        { label: "Permission", value: `Camera permission state: ${permissionState}.`, badge: permissionState, tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral" },
        { label: "Tracked hands", value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible in the frame.", badge: `${trackedHands}`, tone: trackedHands ? "is-good" : "is-neutral" },
        { label: "Visibility coach", value: latestFrameAnalysis?.primaryWarning || "Framing looks usable for template matching.", badge: latestFrameAnalysis?.primaryWarning ? "Adjust" : "Good", tone: latestFrameAnalysis?.primaryWarning ? "is-bad" : "is-good" },
        { label: "Closest match", value: latestPredictions[0] ? `${prettifyLabel(latestPredictions[0].label)} at ${Math.round(latestPredictions[0].score * 100)}%.` : "No frame has been scored yet.", badge: latestPredictions[0] ? "Scored" : "Idle", tone: latestPredictions[0] ? "is-good" : "is-neutral" }
    ];

    if (isAlphabetMode() && getCurrentGesture().id === "A" && latestGeometryDebug) {
        rows.push({
            label: "A geometry",
            value: `Finger bend ${Math.round(latestGeometryDebug.curledScore * 100)}%, thumb ${Math.round(latestGeometryDebug.thumbScore * 100)}%, body ${Math.round(latestGeometryDebug.bodyPositionScore * 100)}%. Allowed mismatch: 60%.`,
            badge: "A debug",
            tone: "is-good"
        });
    }

    diagnosticsSummary.textContent = extra.summary || "This page now uses a last-frame landmark template matcher with a configurable tolerance.";
    renderDiagnosticRows(rows);
}

async function setupMatcher() {
    gestures = buildGestures(getModeConfig().labels);
    currentGestureIndex = Math.min(currentGestureIndex, gestures.length - 1);
    calibrationStatus.textContent = `${getModeConfig().title} template matcher ready.`;
    renderGestureCard();
    await collectDiagnostics();
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
    drawTemplateOverlay(results, getCurrentGesture());
    trackedHands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    latestFrameAnalysis = analyzeCurrentFrame(results);
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % FRAME_SKIP === 0) {
        const predictions = buildPredictions(results);
        renderPredictions(predictions);
        evaluatePredictions(predictions);
        collectDiagnostics().catch(console.error);
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

async function switchMode(nextMode) {
    if (nextMode === currentMode) {
        return;
    }
    currentMode = nextMode;
    renderModeButtons();
    resetTrackingState();
    await setupMatcher();
    renderPredictions([]);
    renderStatus(0, 0, "Move your hand until the live landmarks line up with the target points.");
}

modeWordsBtn.addEventListener("click", () => {
    switchMode("words").catch(console.error);
});

modeAlphabetBtn.addEventListener("click", () => {
    switchMode("alphabet").catch(console.error);
});

nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
calibrateBtn.addEventListener("click", () => setupMatcher().catch(console.error));

window.addEventListener("beforeunload", () => {
    stopLoop();
});

renderModeButtons();
setupMatcher().then(() => {
    renderPredictions([]);
    renderStatus(0, 0, "Move your hand until the live landmarks line up with the target points.");
    return startCamera();
}).catch(console.error);
