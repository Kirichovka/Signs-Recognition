import {
    ALPHABET_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    MAX_SEQUENCE,
    describeCameraError,
    drawHolisticResults,
    featureVectorFromResults,
    getCameraPermissionState,
    loadBrowserModel,
    predictWithBrowserModel,
    prettifyLabel,
    startCameraStream,
    stopMediaStream
} from "./sign-model-runtime.js?v=20260318-8";

const HOLD_SECONDS = 1.0;
const WORD_SCORE_THRESHOLD = 0.45;
const ALPHABET_SCORE_THRESHOLD = 0.6;

const WORD_LABELS = ["HELLO", "BYE", "YES", "NO", "PLEASE", "SORRY", "HELP", "THANKYOU", "WELCOME1", "EAT1", "DRINK1", "WATER", "MOTHER", "FATHER", "FAMILY", "HOME", "HOUSE", "SCHOOL", "WORK", "FRIEND", "LOVE", "WANT1", "NEED", "COME", "COMEHERE", "GO", "STOP", "FINISH", "GOOD", "BAD", "HAPPY", "SAD", "NOW", "MORE", "NOT", "KNOW", "DONTKNOW", "NOTUNDERSTAND", "GOAHEAD", "GREAT"];
const ALPHABET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"];

const MODE_CONFIG = {
    words: {
        title: "Everyday Words",
        summary: "Practice common ASL words and short everyday signs with the temporal browser model.",
        modelName: DEFAULT_MODEL_NAME,
        fallbackLabels: WORD_LABELS,
        targetLabel: "Target Sign",
        tips: "Keep one signer centered, show both hands and upper body when needed, and hold the target sign for about a second so the model sees a stable 40-frame sequence.",
        emptyStatus: "Show the target sign to the camera.",
        waitingStatus: "Hold the target sign steady so the browser model sees a full 40-frame sequence."
    },
    alphabet: {
        title: "Alphabet",
        summary: "Practice single ASL letters with the static browser model. The page switches to the alphabet classifier automatically.",
        modelName: ALPHABET_MODEL_NAME,
        fallbackLabels: ALPHABET_LABELS,
        targetLabel: "Target Letter",
        tips: "Keep one hand clearly visible, center it in the frame, and hold the handshape still for about a second so the alphabet model sees a clean image.",
        emptyStatus: "Show the target letter to the camera.",
        waitingStatus: "Hold the target letter still so the browser model can classify the current frame."
    }
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
let frameBuffer = [];
let predictionInFlight = false;
let frameCounter = 0;
let lastCameraError = "";
let modelState = null;
let cameraReady = false;
let trackedHands = 0;
let latestFrameAnalysis = null;

function getModeConfig() { return MODE_CONFIG[currentMode]; }
function isAlphabetMode() { return currentMode === "alphabet"; }
function isImageModel() { return modelState?.model_type === "image" || isAlphabetMode(); }
function getScoreThreshold() { return isImageModel() ? ALPHABET_SCORE_THRESHOLD : WORD_SCORE_THRESHOLD; }

function buildGestures(labels) {
    return labels.map(label => ({
        id: label,
        title: prettifyLabel(label),
        instruction: isAlphabetMode()
            ? `Shape the hand for "${prettifyLabel(label)}" and keep the letter still until the progress bar fills.`
            : `Show the sign for "${prettifyLabel(label)}" and hold it steady until the progress bar fills.`
    }));
}

function getCurrentGesture() { return gestures[currentGestureIndex] || gestures[0]; }

const GESTURE_ZONE_MAP = { HELLO: "head", BYE: "head", YES: "neutral", NO: "mouth", PLEASE: "chest", SORRY: "chest", HELP: "chest", THANKYOU: "mouth", WELCOME1: "chest", EAT1: "mouth", DRINK1: "mouth", WATER: "mouth", MOTHER: "head", FATHER: "head", FAMILY: "chest", HOME: "neutral", HOUSE: "chest", SCHOOL: "chest", WORK: "chest", FRIEND: "chest", LOVE: "chest", WANT1: "chest", NEED: "chest", COME: "neutral", COMEHERE: "neutral", GO: "neutral", STOP: "chest", FINISH: "chest", GOOD: "mouth", BAD: "mouth", HAPPY: "chest", SAD: "head", NOW: "neutral", MORE: "chest", NOT: "neutral", KNOW: "head", DONTKNOW: "head", NOTUNDERSTAND: "head", GOAHEAD: "neutral", GREAT: "chest" };
const GESTURE_MOVE_MAP = { HELLO: "swing", BYE: "swing", THANKYOU: "right", WELCOME1: "right", EAT1: "short", DRINK1: "short", WATER: "short", COME: "right", COMEHERE: "right", GO: "right", STOP: "tap", FINISH: "right", GOOD: "down", BAD: "down", HAPPY: "up", SAD: "down", MORE: "tap", GOAHEAD: "right", GREAT: "down" };
const TWO_HAND_GESTURES = new Set(["HELP", "WELCOME1", "FAMILY", "HOUSE", "SCHOOL", "WORK", "FRIEND", "LOVE", "MORE", "STOP"]);
const GESTURE_COPY = { HELLO: "Raise the greeting hand near the face and give a clear relaxed wave.", BYE: "Use a short goodbye wave near face level and freeze after the motion finishes.", YES: "Keep the handshape compact and the movement small.", NO: "Keep the sign closer to the mouth area.", PLEASE: "Place the sign in front of the chest and keep the circular motion compact.", SORRY: "Use a chest-level circular motion and finish in the center.", HELP: "Keep both hands visible in front of the chest.", THANKYOU: "Start near the mouth and move the hand outward.", WELCOME1: "Let the movement open outward from the chest.", EAT1: "Bring the hand toward the mouth in a short eating motion.", DRINK1: "Move the hand toward the mouth like a drink action.", WATER: "Keep the water handshape near the mouth.", MOTHER: "Stay near the face so the contact point reads clearly.", FATHER: "Keep the sign near the upper face area.", FAMILY: "Use both hands in front of the chest.", HOME: "Make the sign in neutral space and finish centered.", HOUSE: "Use both hands to outline the shape cleanly.", SCHOOL: "Keep both hands at chest height and make the contact crisp.", WORK: "Use a clear two-hand tapping action.", FRIEND: "Keep both hands visible and make the linking motion neat.", LOVE: "Cross or close the arms in front of the chest and then hold.", WANT1: "Keep the sign chest-high and let the movement settle.", NEED: "Show a decisive closing motion and freeze right after it finishes.", COME: "Let the hand travel toward the body in a clear inviting motion.", COMEHERE: "Use the calling motion toward yourself and keep the hand centered.", GO: "Push the sign outward with a clean directional path.", STOP: "Show the stopping contact clearly.", FINISH: "Use a short finishing motion and hold the final handshape.", GOOD: "Start near the mouth and move downward cleanly.", BAD: "Begin higher and finish lower.", HAPPY: "Keep the sign at chest level and use a smooth upward motion.", SAD: "Begin higher on the face and let the motion drop.", NOW: "Keep the sign compact in neutral space.", MORE: "Use a repeated tap motion evenly.", NOT: "Keep the handshape distinct and avoid drifting away from center.", KNOW: "Start near the forehead and freeze once the handshape is formed.", DONTKNOW: "Use the thought-related motion near the head and keep it compact.", NOTUNDERSTAND: "Stay near the head area so the confusion motion is readable.", GOAHEAD: "Use a clear forward or sideways release gesture and then pause.", GREAT: "Make the sign strong and centered with a clean finishing hold." };
const ALPHABET_HINTS = { A: "Closed fist with the thumb resting along the side.", B: "Flat hand with fingers together and thumb folded across the palm.", C: "Curve the hand into a clear C-shape.", D: "Index finger up, thumb touching the middle finger.", E: "Curl all fingers down toward the thumb.", F: "Index finger and thumb make a small circle, other fingers up.", G: "Index finger and thumb point sideways with a narrow gap.", H: "Index and middle fingers extend sideways together.", I: "Only the pinky is raised.", K: "Make a V-shape and place the thumb at the base of the middle finger.", L: "Index finger up and thumb out to form an L.", M: "Thumb tucked under the first three fingers.", N: "Thumb tucked under the first two fingers.", O: "Curve all fingertips toward the thumb to make an O.", P: "Like K, but angled downward.", Q: "Like G, but angled downward.", R: "Cross the index and middle fingers.", S: "Closed fist with the thumb across the front of the fingers.", T: "Thumb tucked between the index and middle fingers.", U: "Index and middle fingers point up together.", V: "Index and middle fingers point up and spread apart.", W: "Three fingers point up.", X: "Index finger raised in a hooked shape.", Y: "Thumb and pinky extended, middle fingers folded." };

function getGestureTrackingSpec(gesture) {
    if (isAlphabetMode()) {
        return { zone: "neutral", movement: "still", twoHands: false, requiredHands: 1 };
    }
    const twoHands = TWO_HAND_GESTURES.has(gesture.id);
    return { zone: GESTURE_ZONE_MAP[gesture.id] || "neutral", movement: GESTURE_MOVE_MAP[gesture.id] || "still", twoHands, requiredHands: twoHands ? 2 : 1 };
}

function getGestureGuide(gesture) {
    if (isAlphabetMode()) {
        return {
            title: `Letter ${gesture.title}`,
            description: ALPHABET_HINTS[gesture.id] || `Hold the handshape for ${gesture.title} still and centered so the alphabet model sees a clean image.`,
            chips: ["One hand", "Static handshape", "Centered frame"],
            steps: [
                ALPHABET_HINTS[gesture.id] || `Shape the hand for ${gesture.title}.`,
                "Center the hand in front of the camera and keep the wrist still.",
                `Hold the pose for about ${HOLD_SECONDS} second so the alphabet model can lock onto it.`
            ],
            visual: { x: "50%", y: "52%", rotation: 0, scale: 1, movementTone: "Static handshape", isStill: true }
        };
    }

    const spec = getGestureTrackingSpec(gesture);
    const zoneTitles = { head: "Head or face level", mouth: "Near the mouth", chest: "Upper chest space", waist: "Lower torso space", neutral: "Neutral signing space" };
    const zoneDescriptions = { head: "Start the sign around face level and keep your shoulders visible.", mouth: "Bring the sign close to the mouth area without covering the face.", chest: "Keep the sign in front of the upper chest and near the center.", waist: "Drop the hand lower toward the torso.", neutral: "Use the open space in front of your chest and shoulders." };
    const motionDescriptions = { still: "Hold the handshape steady once you form it.", right: "Add a small sideways movement toward your dominant-hand side.", left: "Let the hand travel smoothly across the body line.", up: "Lift the sign in a short upward motion.", down: "Finish with a controlled drop rather than a fast swing.", tap: "Use a short repeated touch or tap motion.", swing: "Rock the hand slightly back and forth instead of freezing it.", short: "Use a short, compact motion and return to the same position." };
    const visual = { head: { x: "50%", y: "28%" }, mouth: { x: "50%", y: "38%" }, chest: { x: "50%", y: "54%" }, waist: { x: "50%", y: "72%" }, neutral: { x: "50%", y: "52%" } }[spec.zone];
    const movementVisual = { still: { rotation: 0, scale: 1, tone: "Hold" }, right: { rotation: 0, scale: 1, tone: "Move right" }, left: { rotation: 180, scale: 1, tone: "Move left" }, up: { rotation: -90, scale: 1, tone: "Lift up" }, down: { rotation: 90, scale: 1, tone: "Move down" }, tap: { rotation: 90, scale: 0.55, tone: "Short tap" }, swing: { rotation: 0, scale: 0.8, tone: "Rock gently" }, short: { rotation: 0, scale: 0.55, tone: "Short move" } }[spec.movement];
    return {
        title: zoneTitles[spec.zone],
        description: GESTURE_COPY[gesture.id] || `Practice the sign for "${gesture.title}" and let the model confirm when the final shape is stable.`,
        chips: [zoneTitles[spec.zone], spec.twoHands ? "Often two hands" : "Usually one main hand", spec.movement === "still" ? "Hold after shaping" : "Includes visible motion"],
        steps: [zoneDescriptions[spec.zone], motionDescriptions[spec.movement], spec.twoHands ? "Keep both hands inside the frame and at roughly the same depth." : "Lead with one clear hand and keep the non-dominant hand quiet.", `Once the sign looks right, freeze the end position for about ${HOLD_SECONDS} second.`],
        visual: { x: visual.x, y: visual.y, rotation: movementVisual.rotation, scale: movementVisual.scale, movementTone: movementVisual.tone, isStill: spec.movement === "still" }
    };
}

function renderModeButtons() {
    modeWordsBtn.classList.toggle("is-active", currentMode === "words");
    modeAlphabetBtn.classList.toggle("is-active", currentMode === "alphabet");
    modeWordsBtn.setAttribute("aria-pressed", String(currentMode === "words"));
    modeAlphabetBtn.setAttribute("aria-pressed", String(currentMode === "alphabet"));
}

function renderGuideCard() {
    const guide = getGestureGuide(getCurrentGesture());
    gestureGuideTitle.textContent = guide.title;
    gestureGuideBadge.textContent = guide.visual.movementTone;
    gestureGuideDescription.textContent = guide.description;
    gestureGuideHand.style.setProperty("--guide-x", guide.visual.x);
    gestureGuideHand.style.setProperty("--guide-y", guide.visual.y);
    gestureGuideArrow.style.setProperty("--guide-x", guide.visual.x);
    gestureGuideArrow.style.setProperty("--guide-y", guide.visual.y);
    gestureGuideArrow.style.setProperty("--guide-rotation", guide.visual.rotation);
    gestureGuideArrow.style.setProperty("--guide-scale", guide.visual.scale);
    gestureGuideArrow.classList.toggle("is-still", guide.visual.isStill);
    gestureGuideChips.innerHTML = "";
    guide.chips.forEach(chipText => {
        const chip = document.createElement("span");
        chip.className = "gesture-guide-chip";
        chip.textContent = chipText;
        gestureGuideChips.appendChild(chip);
    });
    gestureGuideSteps.innerHTML = "";
    guide.steps.forEach((stepText, index) => {
        const row = document.createElement("div");
        row.className = "gesture-guide-step";
        row.innerHTML = `<div class="gesture-guide-step-index">${index + 1}</div><div class="gesture-guide-step-copy">${stepText}</div>`;
        gestureGuideSteps.appendChild(row);
    });
}

function averagePoint(points) {
    if (!points.length) { return null; }
    return { x: points.reduce((sum, point) => sum + point.x, 0) / points.length, y: points.reduce((sum, point) => sum + point.y, 0) / points.length };
}

function visiblePoint(point, minVisibility = 0.2) {
    return !!point && (point.visibility === undefined || point.visibility >= minVisibility);
}

function analyzeCurrentFrame(results) {
    const pose = results.poseLandmarks || [];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const shouldersVisible = visiblePoint(leftShoulder) && visiblePoint(rightShoulder);
    const handSets = [results.leftHandLandmarks || null, results.rightHandLandmarks || null].filter(Boolean);
    const handCentroids = handSets.map(hand => averagePoint(hand)).filter(Boolean);
    const primaryCentroid = averagePoint(handCentroids);
    const warnings = [];

    if (!shouldersVisible) {
        warnings.push("You're not visible enough. Keep your head and both shoulders in frame.");
    }

    if (isAlphabetMode()) {
        if (!handCentroids.length) {
            warnings.push("Show one clear handshape for the target letter.");
        } else if (handCentroids.length > 1) {
            warnings.push("Use one dominant hand for the letter and keep the other hand quieter.");
        } else if (primaryCentroid && (primaryCentroid.x < 0.2 || primaryCentroid.x > 0.8 || primaryCentroid.y < 0.18 || primaryCentroid.y > 0.82)) {
            warnings.push("Move the letter hand closer to the center of the frame.");
        }
        return { handsVisible: handCentroids.length, zoneTarget: "neutral", liveZone: "neutral", zoneOk: handCentroids.length >= 1, primaryWarning: warnings[0] || "" };
    }

    const gesture = getCurrentGesture();
    const spec = getGestureTrackingSpec(gesture);
    const nose = pose[0];
    const mouthLeft = pose[9];
    const mouthRight = pose[10];
    const faceVisible = visiblePoint(nose) || (visiblePoint(mouthLeft) && visiblePoint(mouthRight));
    if (spec.zone === "mouth" && !faceVisible) {
        warnings.push("Your face is not visible enough for a mouth-level sign.");
    }
    if (handCentroids.length < spec.requiredHands) {
        warnings.push(spec.requiredHands === 2 ? "This sign needs both hands visible in the frame." : "Raise your signing hand fully into the frame.");
    }
    if (primaryCentroid && (primaryCentroid.x < 0.12 || primaryCentroid.x > 0.88 || primaryCentroid.y < 0.08 || primaryCentroid.y > 0.92)) {
        warnings.push("Part of the hand is near the edge of the frame. Move back to the center.");
    }
    return { handsVisible: handCentroids.length, zoneTarget: spec.zone, liveZone: spec.zone, zoneOk: !warnings.length, primaryWarning: warnings[0] || "" };
}

function percentToViewBox(value) { return Number.parseFloat(String(value).replace("%", "")) || 50; }

function renderGesturePreview(gesture) {
    if (isAlphabetMode()) {
        gestureEmoji.innerHTML = `<svg viewBox="0 0 100 100" class="gesture-preview-svg" aria-hidden="true"><defs><linearGradient id="alphabetPreviewBg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#2563eb" /><stop offset="100%" stop-color="#38bdf8" /></linearGradient></defs><rect x="2" y="2" width="96" height="96" rx="24" fill="url(#alphabetPreviewBg)" /><circle cx="50" cy="28" r="13" fill="rgba(255,255,255,0.14)" /><rect x="27" y="43" width="46" height="22" rx="10" fill="rgba(255,255,255,0.14)" /><text x="50" y="61" text-anchor="middle" font-size="34" font-weight="800" fill="rgba(255,255,255,0.98)" font-family="Arial, sans-serif">${gesture.title}</text><text x="50" y="84" text-anchor="middle" font-size="10" font-weight="700" fill="rgba(255,255,255,0.78)" font-family="Arial, sans-serif">HOLD STILL</text></svg>`;
        return;
    }
    const guide = getGestureGuide(gesture);
    const spec = getGestureTrackingSpec(gesture);
    const handX = percentToViewBox(guide.visual.x);
    const handY = percentToViewBox(guide.visual.y);
    const secondHandX = spec.twoHands ? Math.max(24, Math.min(76, handX + (handX >= 50 ? -16 : 16))) : handX;
    const secondHandY = spec.twoHands ? Math.min(78, handY + 4) : handY;
    const zoneColor = { head: "#f97316", mouth: "#ef4444", chest: "#2563eb", waist: "#14b8a6", neutral: "#8b5cf6" }[spec.zone] || "#2563eb";
    gestureEmoji.innerHTML = `<svg viewBox="0 0 100 100" class="gesture-preview-svg" aria-hidden="true"><defs><linearGradient id="gesturePreviewBg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#f97316" /><stop offset="100%" stop-color="#fb923c" /></linearGradient></defs><rect x="2" y="2" width="96" height="96" rx="24" fill="url(#gesturePreviewBg)" /><ellipse cx="50" cy="20" rx="12" ry="11" fill="rgba(255,255,255,0.28)" /><rect x="28" y="31" width="44" height="38" rx="16" fill="rgba(255,255,255,0.22)" /><ellipse cx="${handX}" cy="${handY}" rx="14" ry="9" fill="rgba(255,255,255,0.18)" stroke="${zoneColor}" stroke-width="2.5" />${spec.twoHands ? `<ellipse cx="${secondHandX}" cy="${secondHandY}" rx="13" ry="8" fill="rgba(255,255,255,0.16)" stroke="${zoneColor}" stroke-width="2.5" />` : ""}<g transform="translate(${handX} ${handY}) rotate(${guide.visual.rotation}) scale(${guide.visual.scale} 1)" ${guide.visual.isStill ? "opacity=\"0\"" : ""}><line x1="-18" y1="0" x2="10" y2="0" stroke="rgba(255,255,255,0.92)" stroke-width="4" stroke-linecap="round" /><path d="M 10 -7 L 20 0 L 10 7" fill="none" stroke="rgba(255,255,255,0.92)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" /></g><g transform="translate(${handX} ${handY})"><rect x="-10" y="-9" width="20" height="18" rx="7" fill="rgba(255,255,255,0.98)" /><rect x="-10" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.98)" /><rect x="-4.5" y="-21" width="4" height="11" rx="2" fill="rgba(255,255,255,0.98)" /><rect x="1" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.98)" /><rect x="6.5" y="-18" width="4" height="8" rx="2" fill="rgba(255,255,255,0.98)" /><rect x="-14" y="-2" width="7" height="4" rx="2" transform="rotate(-28 -14 -2)" fill="rgba(255,255,255,0.98)" /></g>${spec.twoHands ? `<g transform="translate(${secondHandX} ${secondHandY}) scale(0.88)"><rect x="-10" y="-9" width="20" height="18" rx="7" fill="rgba(255,255,255,0.94)" /><rect x="-10" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.94)" /><rect x="-4.5" y="-21" width="4" height="11" rx="2" fill="rgba(255,255,255,0.94)" /><rect x="1" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.94)" /><rect x="6.5" y="-18" width="4" height="8" rx="2" fill="rgba(255,255,255,0.94)" /><rect x="-14" y="-2" width="7" height="4" rx="2" transform="rotate(-28 -14 -2)" fill="rgba(255,255,255,0.94)" /></g>` : ""}</svg>`;
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
        fingerChecklist.innerHTML = "<div class=\"gesture-check-row\"><span>Waiting</span><span class=\"gesture-check-badge is-neutral\">No predictions yet</span></div>";
        return;
    }
    const target = getCurrentGesture().id;
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        row.innerHTML = `<span>${index + 1}. ${prettifyLabel(item.label)}</span><span class="gesture-check-badge ${item.label === target ? "is-good" : "is-neutral"}">${Math.round(item.score * 100)}%</span>`;
        fingerChecklist.appendChild(row);
    });
}

function resetTrackingState() {
    holdStartedAt = 0;
    lastSuccess = false;
    frameBuffer = [];
    trackedHands = 0;
    latestFrameAnalysis = null;
}

function setGesture(index) {
    currentGestureIndex = (index + gestures.length) % gestures.length;
    resetTrackingState();
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, getModeConfig().emptyStatus);
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
    diagnosticsSummary.textContent = extra.summary || "The trainer is running fully in the browser with ONNX Runtime Web.";
    renderDiagnosticRows([
        { label: "Mode", value: `${getModeConfig().title} using ${getModeConfig().modelName}.`, badge: getModeConfig().title, tone: "is-good" },
        { label: "Browser model", value: modelState ? `Loaded ${modelState.num_classes} labels from ${modelState.model_name}.` : "Model has not loaded yet.", badge: modelState ? (isImageModel() ? "Image" : "Sequence") : "Loading", tone: modelState ? "is-good" : "is-neutral" },
        { label: isAlphabetMode() ? "Target letter" : "Target sign", value: getCurrentGesture().title, badge: `${currentGestureIndex + 1}/${gestures.length}`, tone: "is-good" },
        { label: "Camera state", value: cameraReady ? "Video frames are available for holistic tracking." : "Camera stream is not ready yet.", badge: cameraReady ? "Live" : "Waiting", tone: cameraReady ? "is-good" : "is-neutral" },
        { label: "Permission", value: `Camera permission state: ${permissionState}.`, badge: permissionState, tone: permissionState === "granted" ? "is-good" : permissionState === "denied" ? "is-bad" : "is-neutral" },
        { label: "Tracked hands", value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible in the frame.", badge: `${trackedHands}`, tone: trackedHands ? "is-good" : "is-neutral" },
        { label: isAlphabetMode() ? "Hand framing" : "Target zone", value: latestFrameAnalysis ? (isAlphabetMode() ? `One centered hand is preferred. Visible hands right now: ${latestFrameAnalysis.handsVisible}.` : `Expected: ${latestFrameAnalysis.zoneTarget}.`) : "Waiting for a tracked frame.", badge: latestFrameAnalysis?.zoneOk ? "Aligned" : "Check framing", tone: latestFrameAnalysis?.zoneOk ? "is-good" : "is-neutral" },
        { label: "Visibility coach", value: latestFrameAnalysis?.primaryWarning || "Framing looks usable for recognition.", badge: latestFrameAnalysis?.primaryWarning ? "Adjust" : "Good", tone: latestFrameAnalysis?.primaryWarning ? "is-bad" : "is-good" },
        { label: "Frame buffer", value: isImageModel() ? "Alphabet mode predicts from the current live frame." : `${frameBuffer.length} / ${MAX_SEQUENCE} frames buffered for inference.`, badge: isImageModel() ? "Live" : `${frameBuffer.length}`, tone: isImageModel() || frameBuffer.length >= MAX_SEQUENCE ? "is-good" : "is-neutral" },
        { label: "Last camera error", value: lastCameraError || "No camera error has been recorded in this session.", badge: lastCameraError ? "Has error" : "Clear", tone: lastCameraError ? "is-bad" : "is-good" }
    ]);
}

async function loadModel(forceReload = false) {
    calibrationStatus.textContent = `Loading ${getModeConfig().title.toLowerCase()} model...`;
    try {
        modelState = await loadBrowserModel(getModeConfig().modelName, { forceReload });
        gestures = buildGestures(modelState.label_names || getModeConfig().fallbackLabels);
        currentGestureIndex = Math.min(currentGestureIndex, gestures.length - 1);
        renderGestureCard();
        calibrationStatus.textContent = isImageModel() ? `Alphabet model ready. Loaded ${modelState.num_classes} letters from ${modelState.model_name}.` : `Word model ready. Loaded ${modelState.num_classes} trained signs from ${modelState.model_name}.`;
    } catch (error) {
        console.error(error);
        modelState = null;
        gestures = buildGestures(getModeConfig().fallbackLabels);
        calibrationStatus.textContent = `Could not load browser model: ${error.message}`;
    }
    await collectDiagnostics();
}

function evaluatePredictions(predictions) {
    const target = getCurrentGesture();
    const best = predictions[0] || null;
    const targetPrediction = predictions.find(item => item.label === target.id) || null;
    const targetScore = targetPrediction?.score || 0;
    const frameAnalysis = latestFrameAnalysis;
    let holdProgress = 0;
    let stableSuccess = false;
    let statusText = getModeConfig().waitingStatus;

    if (frameAnalysis?.primaryWarning) {
        holdStartedAt = 0;
        statusText = frameAnalysis.primaryWarning;
    } else if (best && best.label === target.id && best.score >= getScoreThreshold() && frameAnalysis?.zoneOk !== false) {
        if (!holdStartedAt) { holdStartedAt = performance.now(); }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        stableSuccess = elapsed >= HOLD_SECONDS;
        statusText = stableSuccess ? `Matched ${target.title}. Moving to the next ${isAlphabetMode() ? "letter" : "sign"}.` : `Matched ${target.title}. Keep holding it a little longer.`;
    } else if (targetPrediction && targetPrediction.score >= 0.28) {
        holdStartedAt = 0;
        statusText = `The model weakly sees ${target.title} (${Math.round(targetPrediction.score * 100)}%). Clean up the framing and hold more steadily.`;
    } else if (best) {
        holdStartedAt = 0;
        statusText = best.score < 0.28 ? "The model is not confident yet. Re-center yourself and make the target larger and clearer." : `Top guess is ${prettifyLabel(best.label)} at ${Math.round(best.score * 100)}%. Adjust toward ${target.title}.`;
    }

    renderStatus(targetScore, holdProgress, statusText);
    if (stableSuccess && !lastSuccess) {
        lastSuccess = true;
        window.setTimeout(() => setGesture(currentGestureIndex + 1), 450);
        return;
    }
    lastSuccess = stableSuccess;
}

async function predictCurrentInput() {
    const waitingForSequence = !isImageModel() && frameBuffer.length < MAX_SEQUENCE;
    if (predictionInFlight || !modelState || waitingForSequence) { return; }
    predictionInFlight = true;
    try {
        const result = await predictWithBrowserModel(modelState, isImageModel() ? inputVideo : frameBuffer);
        renderPredictions(result.predictions);
        evaluatePredictions(result.predictions);
    } catch (error) {
        console.error(error);
        renderStatus(0, 0, `Prediction failed: ${error.message}`);
    } finally {
        predictionInFlight = false;
        await collectDiagnostics();
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

function onHolisticResults(results) {
    drawHolisticResults(canvasCtx, outputCanvas, inputVideo, results);
    if (!isImageModel()) {
        frameBuffer.push(featureVectorFromResults(results));
        if (frameBuffer.length > MAX_SEQUENCE) { frameBuffer.shift(); }
    }
    trackedHands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    latestFrameAnalysis = analyzeCurrentFrame(results);
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % 6 === 0) {
        predictCurrentInput().catch(console.error);
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
    if (nextMode === currentMode) { return; }
    currentMode = nextMode;
    renderModeButtons();
    resetTrackingState();
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, getModeConfig().emptyStatus);
    await loadModel(false);
}

modeWordsBtn.addEventListener("click", () => { switchMode("words").catch(console.error); });
modeAlphabetBtn.addEventListener("click", () => { switchMode("alphabet").catch(console.error); });
nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
calibrateBtn.addEventListener("click", () => loadModel(true).catch(console.error));

window.addEventListener("beforeunload", () => { stopLoop(); });

renderModeButtons();
gestures = buildGestures(getModeConfig().fallbackLabels);
renderGestureCard();
renderPredictions([]);
renderStatus(0, 0, getModeConfig().emptyStatus);
loadModel(false).then(() => startCamera()).catch(console.error);
