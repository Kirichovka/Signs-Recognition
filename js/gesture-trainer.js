import {
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
} from "./sign-model-runtime.js";

const HOLD_SECONDS = 1.0;
const SCORE_THRESHOLD = 0.45;
const FALLBACK_LABELS = [
    "HELLO", "BYE", "YES", "NO", "PLEASE", "SORRY", "HELP", "THANKYOU", "WELCOME1", "EAT1",
    "DRINK1", "WATER", "MOTHER", "FATHER", "FAMILY", "HOME", "HOUSE", "SCHOOL", "WORK", "FRIEND",
    "LOVE", "WANT1", "NEED", "COME", "COMEHERE", "GO", "STOP", "FINISH", "GOOD", "BAD",
    "HAPPY", "SAD", "NOW", "MORE", "NOT", "KNOW", "DONTKNOW", "NOTUNDERSTAND", "GOAHEAD", "GREAT"
];
const GESTURE_ZONE_MAP = {
    HELLO: "head",
    BYE: "head",
    YES: "neutral",
    NO: "mouth",
    PLEASE: "chest",
    SORRY: "chest",
    HELP: "chest",
    THANKYOU: "mouth",
    WELCOME1: "chest",
    EAT1: "mouth",
    DRINK1: "mouth",
    WATER: "mouth",
    MOTHER: "head",
    FATHER: "head",
    FAMILY: "chest",
    HOME: "neutral",
    HOUSE: "chest",
    SCHOOL: "chest",
    WORK: "chest",
    FRIEND: "chest",
    LOVE: "chest",
    WANT1: "chest",
    NEED: "chest",
    COME: "neutral",
    COMEHERE: "neutral",
    GO: "neutral",
    STOP: "chest",
    FINISH: "chest",
    GOOD: "mouth",
    BAD: "mouth",
    HAPPY: "chest",
    SAD: "head",
    NOW: "neutral",
    MORE: "chest",
    NOT: "neutral",
    KNOW: "head",
    DONTKNOW: "head",
    NOTUNDERSTAND: "head",
    GOAHEAD: "neutral",
    GREAT: "chest"
};
const GESTURE_MOVE_MAP = {
    HELLO: "swing",
    BYE: "swing",
    THANKYOU: "right",
    WELCOME1: "right",
    EAT1: "short",
    DRINK1: "short",
    WATER: "short",
    COME: "right",
    COMEHERE: "right",
    GO: "right",
    STOP: "tap",
    FINISH: "right",
    GOOD: "down",
    BAD: "down",
    HAPPY: "up",
    SAD: "down",
    MORE: "tap",
    GOAHEAD: "right",
    GREAT: "down"
};
const TWO_HAND_GESTURES = new Set([
    "HELP",
    "WELCOME1",
    "FAMILY",
    "HOUSE",
    "SCHOOL",
    "WORK",
    "FRIEND",
    "LOVE",
    "MORE",
    "STOP"
]);
const GESTURE_COPY = {
    HELLO: `Raise the greeting hand near the side of the face and give a clear relaxed wave.`,
    BYE: `Use a short goodbye wave near face level and freeze after the motion finishes.`,
    YES: `Keep the handshape compact and make the movement small so it stays readable.`,
    NO: `Keep the sign closer to the mouth area and avoid dropping it too low.`,
    PLEASE: `Place the sign in front of the chest and keep the circular motion compact.`,
    SORRY: `Use a chest-level circular motion and finish cleanly in the center of the body.`,
    HELP: `Keep both hands visible in front of the chest so the support hand is easy to read.`,
    THANKYOU: `Start near the mouth and move the hand outward without covering the face.`,
    WELCOME1: `Use both hands if natural and let the movement open outward from the chest.`,
    EAT1: `Bring the hand toward the mouth in a short eating motion and hold the end shape.`,
    DRINK1: `Move the hand toward the mouth like a drink action, then pause clearly.`,
    WATER: `Keep the water handshape near the mouth and avoid large travel afterward.`,
    MOTHER: `Stay near the face so the contact point reads clearly.`,
    FATHER: `Keep the sign near the upper face area and hold the final position.`,
    FAMILY: `Use both hands in front of the chest and keep the shape centered.`,
    HOME: `Make the sign in neutral space and finish in a stable centered position.`,
    HOUSE: `Use both hands to outline the shape cleanly in front of the chest.`,
    SCHOOL: `Keep both hands at chest height and make the contact crisp.`,
    WORK: `Use a clear two-hand tapping action in front of the torso.`,
    FRIEND: `Keep both hands visible and make the linking motion neat and compact.`,
    LOVE: `Cross or close the arms/hands in front of the chest and then hold.`,
    WANT1: `Keep the sign chest-high and let the movement settle into the final shape.`,
    NEED: `Show a decisive closing motion and freeze right after it finishes.`,
    COME: `Let the hand travel toward the body in a clear inviting motion.`,
    COMEHERE: `Use the calling motion toward yourself and keep the hand centered.`,
    GO: `Push the sign outward with a clean directional path.`,
    STOP: `Show the stopping contact clearly with both hands visible if you use them.`,
    FINISH: `Use a short finishing motion and hold the final handshape right away.`,
    GOOD: `Start near the mouth and move downward cleanly without rushing.`,
    BAD: `Begin higher and finish lower so the down direction is obvious.`,
    HAPPY: `Keep the sign at chest level and use a smooth positive upward motion.`,
    SAD: `Begin higher on the face and let the motion drop in a controlled way.`,
    NOW: `Keep the sign compact in neutral space and hold it right after forming it.`,
    MORE: `Use both hands if natural and make the repeated tap motion even.`,
    NOT: `Keep the handshape distinct and avoid drifting away from center.`,
    KNOW: `Start near the forehead and freeze once the ending handshape is formed.`,
    DONTKNOW: `Use the thought-related motion near the head and keep it compact.`,
    NOTUNDERSTAND: `Stay near the head area so the confusion/understanding motion is readable.`,
    GOAHEAD: `Use a clear forward or sideways release gesture and then pause.`,
    GREAT: `Make the sign strong and centered with a clean finishing hold.`
};

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
let modelState = null;
let cameraReady = false;
let trackedHands = 0;
let latestFrameAnalysis = null;

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

function getCurrentGesture() {
    return gestures[currentGestureIndex] || gestures[0];
}

function getGestureTrackingSpec(gesture) {
    const twoHands = TWO_HAND_GESTURES.has(gesture.id);
    return {
        zone: GESTURE_ZONE_MAP[gesture.id] || "neutral",
        movement: GESTURE_MOVE_MAP[gesture.id] || "still",
        twoHands,
        requiredHands: twoHands ? 2 : 1
    };
}

function getGestureGuide(gesture) {
    const label = gesture.id;
    const title = gesture.title;

    const spec = getGestureTrackingSpec(gesture);
    const zone = spec.zone;
    const movement = spec.movement;
    const twoHands = spec.twoHands;

    const zoneTitles = {
        head: "Head or face level",
        mouth: "Near the mouth",
        chest: "Upper chest space",
        waist: "Lower torso space",
        neutral: "Neutral signing space"
    };
    const zoneDescriptions = {
        head: `Start the sign around face level and keep your shoulders visible so the model can see where the motion begins.`,
        mouth: `Bring the sign close to the mouth area, but keep the hand separated enough that the camera still sees the hand shape clearly.`,
        chest: `Keep the sign in front of the upper chest, not too low, and avoid drifting outside the camera center.`,
        waist: `Drop the hand a bit lower toward the torso so the gesture reads closer to belt or hip level.`,
        neutral: `Use the open space in front of your chest and shoulders, with the signer centered and the elbows relaxed.`
    };

    const motionDescriptions = {
        still: "Hold the handshape steady once you form it.",
        right: "Add a small sideways movement toward your dominant-hand side.",
        left: "Let the hand travel smoothly across the body line.",
        up: "Lift the sign in a short upward motion.",
        down: "Finish with a controlled drop rather than a fast swing.",
        tap: "Use a short repeated touch or tap motion.",
        swing: "Rock the hand slightly back and forth instead of freezing it.",
        bounce: "Use a compact bouncing rhythm rather than one single frozen pose.",
        float: "Keep the hand light and drifting rather than rigid.",
        short: "Use a short, compact motion and return to the same position."
    };

    const labelSpecific = GESTURE_COPY;

    const chips = [
        zoneTitles[zone],
        twoHands ? "Often two hands" : "Usually one main hand",
        movement === "still" ? "Hold after shaping" : "Includes visible motion"
    ];

    const steps = [
        zoneDescriptions[zone],
        motionDescriptions[movement],
        twoHands
            ? "Keep both hands inside the frame and at roughly the same depth so the model keeps tracking them."
            : "Lead with one clear hand and keep the non-dominant hand quiet unless the sign needs both hands.",
        `Once the sign looks right, freeze the end position for about ${HOLD_SECONDS} second so the progress bar can fill.`
    ];

    const visual = {
        head: { x: "50%", y: "28%" },
        mouth: { x: "50%", y: "38%" },
        chest: { x: "50%", y: "54%" },
        waist: { x: "50%", y: "72%" },
        neutral: { x: "50%", y: "52%" }
    }[zone];

    const movementVisual = {
        still: { rotation: 0, scale: 1, tone: "Hold" },
        right: { rotation: 0, scale: 1, tone: "Move right" },
        left: { rotation: 180, scale: 1, tone: "Move left" },
        up: { rotation: -90, scale: 1, tone: "Lift up" },
        down: { rotation: 90, scale: 1, tone: "Move down" },
        tap: { rotation: 90, scale: 0.55, tone: "Short tap" },
        swing: { rotation: 0, scale: 0.8, tone: "Rock gently" },
        bounce: { rotation: 90, scale: 0.8, tone: "Bounce" },
        float: { rotation: -35, scale: 0.85, tone: "Float" },
        short: { rotation: 0, scale: 0.55, tone: "Short move" }
    }[movement];

    return {
        title: zoneTitles[zone],
        description: labelSpecific[label] || `Practice the sign for "${title}" in ${zoneTitles[zone].toLowerCase()} and let the model confirm when the final shape is stable.`,
        chips,
        steps,
        visual: {
            x: visual.x,
            y: visual.y,
            rotation: movementVisual.rotation,
            scale: movementVisual.scale,
            movementTone: movementVisual.tone,
            isStill: movement === "still"
        }
    };
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

        const stepIndex = document.createElement("div");
        stepIndex.className = "gesture-guide-step-index";
        stepIndex.textContent = `${index + 1}`;

        const stepCopy = document.createElement("div");
        stepCopy.className = "gesture-guide-step-copy";
        stepCopy.textContent = stepText;

        row.append(stepIndex, stepCopy);
        gestureGuideSteps.appendChild(row);
    });
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

function analyzeCurrentFrame(results) {
    const gesture = getCurrentGesture();
    const spec = getGestureTrackingSpec(gesture);
    const pose = results.poseLandmarks || [];
    const nose = pose[0];
    const mouthLeft = pose[9];
    const mouthRight = pose[10];
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const leftHip = pose[23];
    const rightHip = pose[24];
    const shouldersVisible = visiblePoint(leftShoulder) && visiblePoint(rightShoulder);
    const hipsVisible = visiblePoint(leftHip) && visiblePoint(rightHip);
    const faceVisible = visiblePoint(nose) || (visiblePoint(mouthLeft) && visiblePoint(mouthRight));

    const handSets = [results.leftHandLandmarks || null, results.rightHandLandmarks || null].filter(Boolean);
    const handCentroids = handSets.map(hand => averagePoint(hand)).filter(Boolean);
    const primaryCentroid = averagePoint(handCentroids);

    let shoulderSpan = shouldersVisible
        ? Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y)
        : 0;
    shoulderSpan = Math.max(shoulderSpan, 0.08);

    const shoulderCenter = shouldersVisible
        ? averagePoint([leftShoulder, rightShoulder])
        : { x: 0.5, y: 0.45 };
    const mouthCenter = visiblePoint(mouthLeft) && visiblePoint(mouthRight)
        ? averagePoint([mouthLeft, mouthRight])
        : visiblePoint(nose)
            ? { x: nose.x, y: nose.y + shoulderSpan * 0.12 }
            : { x: shoulderCenter.x, y: shoulderCenter.y - shoulderSpan * 0.55 };
    const hipCenter = hipsVisible
        ? averagePoint([leftHip, rightHip])
        : { x: shoulderCenter.x, y: shoulderCenter.y + shoulderSpan * 1.45 };

    const zoneCenters = {
        head: { x: shoulderCenter.x, y: shoulderCenter.y - shoulderSpan * 0.72 },
        mouth: mouthCenter,
        chest: { x: shoulderCenter.x, y: shoulderCenter.y + shoulderSpan * 0.25 },
        waist: { x: hipCenter.x, y: hipCenter.y - shoulderSpan * 0.12 },
        neutral: { x: shoulderCenter.x, y: shoulderCenter.y + shoulderSpan * 0.35 }
    };

    const warnings = [];
    if (!shouldersVisible) {
        warnings.push("You're not visible enough. Keep your head and both shoulders in frame.");
    }
    if (spec.zone === "mouth" && !faceVisible) {
        warnings.push("Your face is not visible enough for a mouth-level sign.");
    }
    if (handCentroids.length < spec.requiredHands) {
        warnings.push(spec.requiredHands === 2
            ? "This sign needs both hands visible in the frame."
            : "Raise your signing hand fully into the frame.");
    }
    if (shouldersVisible && shoulderSpan < 0.14) {
        warnings.push("Move a little closer to the camera so the model can read your upper body.");
    }
    if (primaryCentroid && (primaryCentroid.x < 0.12 || primaryCentroid.x > 0.88 || primaryCentroid.y < 0.08 || primaryCentroid.y > 0.92)) {
        warnings.push("Part of the hand is near the edge of the frame. Move back to the center.");
    }

    let zoneOk = true;
    let liveZone = "unknown";
    if (primaryCentroid && shouldersVisible) {
        const zoneDistances = Object.entries(zoneCenters).map(([zoneName, center]) => ({
            zoneName,
            distance: Math.hypot(primaryCentroid.x - center.x, primaryCentroid.y - center.y)
        })).sort((a, b) => a.distance - b.distance);
        liveZone = zoneDistances[0]?.zoneName || "unknown";

        if (spec.zone !== "neutral") {
            const expectedCenter = zoneCenters[spec.zone];
            const distanceToExpected = Math.hypot(primaryCentroid.x - expectedCenter.x, primaryCentroid.y - expectedCenter.y);
            zoneOk = distanceToExpected <= shoulderSpan * 0.65;
            if (!zoneOk) {
                const zoneLabels = {
                    head: "higher, near the face",
                    mouth: "closer to the mouth",
                    chest: "closer to the upper chest",
                    waist: "lower, around the torso",
                    neutral: "in the neutral signing space"
                };
                warnings.push(`The sign is in the wrong zone. Move it ${zoneLabels[spec.zone]}.`);
            }
        }
    }

    return {
        gestureId: gesture.id,
        requiredHands: spec.requiredHands,
        handsVisible: handCentroids.length,
        shouldersVisible,
        faceVisible,
        zoneTarget: spec.zone,
        liveZone,
        zoneOk,
        primaryWarning: warnings[0] || "",
        warnings
    };
}

function percentToViewBox(value) {
    return Number.parseFloat(String(value).replace("%", "")) || 50;
}

function renderGesturePreview(gesture) {
    const guide = getGestureGuide(gesture);
    const spec = getGestureTrackingSpec(gesture);
    const handX = percentToViewBox(guide.visual.x);
    const handY = percentToViewBox(guide.visual.y);
    const secondHandX = spec.twoHands ? Math.max(24, Math.min(76, handX + (handX >= 50 ? -16 : 16))) : handX;
    const secondHandY = spec.twoHands ? Math.min(78, handY + 4) : handY;
    const zoneColorMap = {
        head: "#f97316",
        mouth: "#ef4444",
        chest: "#2563eb",
        waist: "#14b8a6",
        neutral: "#8b5cf6"
    };
    const zoneColor = zoneColorMap[spec.zone] || "#2563eb";

    gestureEmoji.innerHTML = `
        <svg viewBox="0 0 100 100" class="gesture-preview-svg" aria-hidden="true">
            <defs>
                <linearGradient id="gesturePreviewBg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#f97316" />
                    <stop offset="100%" stop-color="#fb923c" />
                </linearGradient>
            </defs>
            <rect x="2" y="2" width="96" height="96" rx="24" fill="url(#gesturePreviewBg)" />
            <ellipse cx="50" cy="20" rx="12" ry="11" fill="rgba(255,255,255,0.28)" />
            <rect x="28" y="31" width="44" height="38" rx="16" fill="rgba(255,255,255,0.22)" />
            <ellipse cx="${handX}" cy="${handY}" rx="14" ry="9" fill="rgba(255,255,255,0.18)" stroke="${zoneColor}" stroke-width="2.5" />
            ${spec.twoHands ? `<ellipse cx="${secondHandX}" cy="${secondHandY}" rx="13" ry="8" fill="rgba(255,255,255,0.16)" stroke="${zoneColor}" stroke-width="2.5" />` : ""}
            <g transform="translate(${handX} ${handY}) rotate(${guide.visual.rotation}) scale(${guide.visual.scale} 1)" ${guide.visual.isStill ? 'opacity="0"' : ""}>
                <line x1="-18" y1="0" x2="10" y2="0" stroke="rgba(255,255,255,0.92)" stroke-width="4" stroke-linecap="round" />
                <path d="M 10 -7 L 20 0 L 10 7" fill="none" stroke="rgba(255,255,255,0.92)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
            </g>
            <g transform="translate(${handX} ${handY})">
                <rect x="-10" y="-9" width="20" height="18" rx="7" fill="rgba(255,255,255,0.98)" />
                <rect x="-10" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.98)" />
                <rect x="-4.5" y="-21" width="4" height="11" rx="2" fill="rgba(255,255,255,0.98)" />
                <rect x="1" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.98)" />
                <rect x="6.5" y="-18" width="4" height="8" rx="2" fill="rgba(255,255,255,0.98)" />
                <rect x="-14" y="-2" width="7" height="4" rx="2" transform="rotate(-28 -14 -2)" fill="rgba(255,255,255,0.98)" />
            </g>
            ${spec.twoHands ? `
            <g transform="translate(${secondHandX} ${secondHandY}) scale(0.88)">
                <rect x="-10" y="-9" width="20" height="18" rx="7" fill="rgba(255,255,255,0.94)" />
                <rect x="-10" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.94)" />
                <rect x="-4.5" y="-21" width="4" height="11" rx="2" fill="rgba(255,255,255,0.94)" />
                <rect x="1" y="-20" width="4" height="10" rx="2" fill="rgba(255,255,255,0.94)" />
                <rect x="6.5" y="-18" width="4" height="8" rx="2" fill="rgba(255,255,255,0.94)" />
                <rect x="-14" y="-2" width="7" height="4" rx="2" transform="rotate(-28 -14 -2)" fill="rgba(255,255,255,0.94)" />
            </g>` : ""}
        </svg>
    `;
}

function renderGestureCard() {
    const gesture = getCurrentGesture();
    gestureTitle.textContent = gesture.title;
    gestureInstruction.textContent = gesture.instruction;
    renderGesturePreview(gesture);
    gestureStep.textContent = `${currentGestureIndex + 1} / ${gestures.length}`;
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
        const empty = document.createElement("div");
        empty.className = "gesture-check-row";
        empty.innerHTML = '<span>Waiting</span><span class="gesture-check-badge is-neutral">No predictions yet</span>';
        fingerChecklist.appendChild(empty);
        return;
    }

    const target = getCurrentGesture().id;
    predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "gesture-check-row";
        const label = document.createElement("span");
        label.textContent = `${index + 1}. ${prettifyLabel(item.label)}`;
        const value = document.createElement("span");
        value.textContent = `${Math.round(item.score * 100)}%`;
        const isTarget = item.label === target;
        value.className = `gesture-check-badge ${isTarget ? "is-good" : index === 0 ? "is-neutral" : "is-neutral"}`;
        row.append(label, value);
        fingerChecklist.appendChild(row);
    });
}

function setGesture(index) {
    currentGestureIndex = (index + gestures.length) % gestures.length;
    holdStartedAt = 0;
    lastSuccess = false;
    renderGestureCard();
    renderPredictions([]);
    renderStatus(0, 0, "Show the target sign to the camera.");
}

function renderDiagnosticRows(rows) {
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

async function collectDiagnostics(extra = {}) {
    const permissionState = await getCameraPermissionState();
    diagnosticsSummary.textContent = extra.summary || "The trainer is running fully in the browser with ONNX Runtime Web.";
    renderDiagnosticRows([
        {
            label: "Browser model",
            value: modelState ? `Loaded ${modelState.num_classes} classes from ${modelState.model_name}.` : "Model has not loaded yet.",
            badge: modelState ? "Ready" : "Loading",
            tone: modelState ? "is-good" : "is-neutral"
        },
        {
            label: "Target sign",
            value: getCurrentGesture().title,
            badge: `${currentGestureIndex + 1}/${gestures.length}`,
            tone: "is-good"
        },
        {
            label: "Camera state",
            value: cameraReady ? "Video frames are available for holistic tracking." : "Camera stream is not ready yet.",
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
            value: trackedHands ? `Detected ${trackedHands} hand(s) in the current frame.` : "No hands are currently visible in the frame.",
            badge: `${trackedHands}`,
            tone: trackedHands ? "is-good" : "is-neutral"
        },
        {
            label: "Target zone",
            value: latestFrameAnalysis
                ? `Expected: ${latestFrameAnalysis.zoneTarget}. Live: ${latestFrameAnalysis.liveZone}.`
                : "Waiting for a tracked frame.",
            badge: latestFrameAnalysis?.zoneOk ? "Aligned" : "Check zone",
            tone: latestFrameAnalysis?.zoneOk ? "is-good" : "is-neutral"
        },
        {
            label: "Visibility coach",
            value: latestFrameAnalysis?.primaryWarning || "Framing looks usable for recognition.",
            badge: latestFrameAnalysis?.primaryWarning ? "Adjust" : "Good",
            tone: latestFrameAnalysis?.primaryWarning ? "is-bad" : "is-good"
        },
        {
            label: "Frame buffer",
            value: `${frameBuffer.length} / ${MAX_SEQUENCE} frames buffered for inference.`,
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

async function loadModel() {
    calibrationStatus.textContent = "Loading browser model...";
    try {
        modelState = await loadBrowserModel();
        gestures = buildGestures(modelState.label_names);
        currentGestureIndex = Math.min(currentGestureIndex, gestures.length - 1);
        renderGestureCard();
        calibrationStatus.textContent = `Browser model ready. Loaded ${modelState.num_classes} trained signs from ${modelState.model_name}.`;
    } catch (error) {
        console.error(error);
        modelState = null;
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
    let statusText = "Hold the target sign steady so the browser model sees a full 40-frame sequence.";

    if (frameAnalysis?.primaryWarning) {
        holdStartedAt = 0;
        statusText = frameAnalysis.primaryWarning;
    } else if (best && best.label === target.id && best.score >= SCORE_THRESHOLD && frameAnalysis?.zoneOk !== false) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }
        const elapsed = (performance.now() - holdStartedAt) / 1000;
        holdProgress = Math.min(1, elapsed / HOLD_SECONDS);
        stableSuccess = elapsed >= HOLD_SECONDS;
        statusText = stableSuccess
            ? `Matched ${target.title}. Moving to the next trained sign.`
            : `Matched ${target.title}. Keep holding it a little longer.`;
    } else if (targetPrediction && targetPrediction.score >= 0.28) {
        holdStartedAt = 0;
        statusText = `The model weakly sees ${target.title} (${Math.round(targetPrediction.score * 100)}%). Clean up the framing and hold more steadily.`;
    } else if (best) {
        holdStartedAt = 0;
        statusText = best.score < 0.28
            ? "The model is not confident yet. Re-center yourself and make the sign larger and clearer."
            : `Top guess is ${prettifyLabel(best.label)} at ${Math.round(best.score * 100)}%. Adjust toward ${target.title}.`;
    } else {
        holdStartedAt = 0;
    }

    renderStatus(targetScore, holdProgress, statusText);

    if (stableSuccess && !lastSuccess) {
        lastSuccess = true;
        window.setTimeout(() => setGesture(currentGestureIndex + 1), 450);
        return;
    }
    lastSuccess = stableSuccess;
}

async function predictSequence() {
    if (predictionInFlight || frameBuffer.length < MAX_SEQUENCE || !modelState) {
        return;
    }
    predictionInFlight = true;
    try {
        const result = await predictWithBrowserModel(modelState, frameBuffer);
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
    frameBuffer.push(featureVectorFromResults(results));
    if (frameBuffer.length > MAX_SEQUENCE) {
        frameBuffer.shift();
    }
    trackedHands = [results.leftHandLandmarks, results.rightHandLandmarks].filter(Boolean).length;
    latestFrameAnalysis = analyzeCurrentFrame(results);
    frameCounter += 1;
    cameraReady = true;
    cameraState.textContent = "Camera is live";
    if (frameCounter % 6 === 0) {
        predictSequence().catch(console.error);
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

nextGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex + 1));
prevGestureBtn.addEventListener("click", () => setGesture(currentGestureIndex - 1));
retryCameraBtn.addEventListener("click", () => startCamera().catch(console.error));
refreshDiagnosticsBtn.addEventListener("click", () => collectDiagnostics().catch(console.error));
calibrateBtn.addEventListener("click", () => loadModel().catch(console.error));

window.addEventListener("beforeunload", () => {
    stopLoop();
});

renderGestureCard();
renderPredictions([]);
renderStatus(0, 0, "Show the target sign to the camera.");
loadModel().then(() => startCamera()).catch(console.error);
