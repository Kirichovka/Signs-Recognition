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
const MATCH_THRESHOLD = 0.55;
const HOLD_SECONDS = 0.8;
const PREFERRED_WORD_ORDER = ["hello", "thanks", "yes", "no"];
const AVAILABLE_MEDIA_ASSETS = new Set();
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];
const DRAW_POINT_INDICES = [0, 4, 8, 12, 20];

const signCategories = [
    {
        name: "Greetings & Basics",
        icon: "\u{1F44B}",
        words: [
            { word: "HELLO", icon: "\u{1F44B}", modelLabel: "HELLO", video: "videos/hello.mp4", desc: "Raise your dominant hand near your forehead and move it outward.", aliases: ["hello"] },
            { word: "BYE", icon: "\u{1F44B}", modelLabel: "BYE", video: "videos/bye.mp4", desc: "Open your hand and wave side to side." },
            { word: "PLEASE", icon: "\u{1F970}", modelLabel: "PLEASE", video: "videos/please.mp4", desc: "Place your flat hand on your chest and move it in a circular motion." },
            { word: "THANK YOU", icon: "\u{1F64F}", modelLabel: "THANKYOU", video: "videos/thankyou.mp4", desc: "Touch your fingers to your chin, then move your hand forward.", aliases: ["thanks"] },
            { word: "SORRY", icon: "\u{1F614}", modelLabel: "SORRY", video: "videos/sorry.mp4", desc: "Make a fist and rub it in a circular motion on your chest." },
            { word: "YES", icon: "\u{1F44D}", modelLabel: "YES", video: "videos/yes.mp4", desc: "Make a fist and nod it up and down.", aliases: ["yes"] },
            { word: "NO", icon: "\u{1F44E}", modelLabel: "NO", video: "videos/no.mp4", desc: "Pinch your index and middle fingers to your thumb.", aliases: ["no"] },
            { word: "WATER", icon: "\u{1F4A7}", modelLabel: "WATER", video: "videos/water.mp4", desc: "Tap your index finger on your chin in a W shape." }
        ]
    },
    {
        name: "Feelings & Responses",
        icon: "\u{1F60A}",
        words: [
            { word: "GOOD", icon: "\u2728", modelLabel: "GOOD", video: "videos/good.gif", desc: "Touch your fingers to your lips, then move your hand down to your other palm." },
            { word: "BAD", icon: "\u{1F44E}", modelLabel: "BAD", video: "videos/bad.mp4", desc: "Touch your fingers to your lips, then move your hand downward and flip it over." },
            { word: "GREAT", icon: "\u{1F603}", modelLabel: "GREAT", video: "videos/great.mp4", desc: "Start with both hands up, then move them down emphatically." },
            { word: "HAPPY", icon: "\u{1F60A}", modelLabel: "HAPPY", video: "videos/happy.mp4", desc: "Brush your chest upward with both hands." },
            { word: "SAD", icon: "\u{1F622}", modelLabel: "SAD", video: "videos/sad.mp4", desc: "Bring your hands down your face to show a sad expression." },
            { word: "LOVE", icon: "\u2764\uFE0F", modelLabel: "LOVE", video: "videos/love.mp4", desc: "Cross your arms over your chest like a hug." }
        ]
    },
    {
        name: "Daily Actions",
        icon: "\u{1F3C3}",
        words: [
            { word: "EAT", icon: "\u{1F34E}", modelLabel: "EAT1", video: "videos/eat.mp4", desc: "Bring your fingers together and tap your mouth." },
            { word: "DRINK", icon: "\u{1F964}", modelLabel: "DRINK1", video: "videos/drink.mp4", desc: "Pretend to hold a cup and tilt it toward your mouth." },
            { word: "HELP", icon: "\u{1F198}", modelLabel: "HELP", video: "videos/help.mp4", desc: "Place one hand flat, put the other hand with a thumbs-up on top, then lift." },
            { word: "WORK", icon: "\u{1F528}", modelLabel: "WORK", video: "videos/work.mp4", desc: "Tap one fist on top of the other a couple of times." },
            { word: "STOP", icon: "\u{1F6D1}", modelLabel: "STOP", video: "videos/stop.mp4", desc: "Bring one hand down sharply onto the palm of the other." },
            { word: "GO", icon: "\u{1F3C3}", modelLabel: "GO", video: "videos/go.mp4", desc: "Point forward with both hands and move them outward." },
            { word: "GO AHEAD", icon: "\u27A1\uFE0F", modelLabel: "GOAHEAD", video: "videos/go-ahead.mp4", desc: "Move both hands forward in a gentle pushing motion." },
            { word: "FINISH", icon: "\u2705", modelLabel: "FINISH", video: "videos/finish.mp4", desc: "Twist both hands outward quickly from palms-in to palms-out." },
            { word: "COME", icon: "\u{1F44B}", modelLabel: "COME", video: "videos/come.mp4", desc: "Palm up, curl fingers toward yourself." },
            { word: "WANT", icon: "\u{1F932}", modelLabel: "WANT1", video: "videos/want.mp4", desc: "Hands open, palms up, pull them slightly toward yourself while closing fingers." },
            { word: "NEED", icon: "\u2757", modelLabel: "NEED", video: "videos/need.mp4", desc: "Make fists and pull them downward sharply." }
        ]
    },
    {
        name: "Social & Home",
        icon: "\u{1F46A}",
        words: [
            { word: "MOTHER", icon: "\u{1F469}", modelLabel: "MOTHER", video: "videos/mother.mp4", desc: "Spread your fingers and touch your thumb to your chin." },
            { word: "HOME", icon: "\u{1F3E0}", modelLabel: "HOME", video: "videos/home.mp4", desc: "Bring your fingers from the side of your mouth to the side of your cheek." },
            { word: "FAMILY", icon: "\u{1F46A}", modelLabel: "FAMILY", video: "videos/family.mp4", desc: "Form F handshapes with both hands and make a circle outward." },
            { word: "FRIEND", icon: "\u{1F91D}", modelLabel: "FRIEND", video: "videos/friend.mp4", desc: "Hook your index fingers together, then switch them." },
            { word: "WELCOME", icon: "\u{1F917}", modelLabel: "WELCOME1", video: "videos/welcome.jpg", desc: "Start with both hands in front of you, palms up, then bring them toward your chest." },
            { word: "DON'T UNDERSTAND", icon: "\u{1F914}", modelLabel: "NOTUNDERSTAND", video: "videos/dont-understand.mp4", desc: "Make a confused face and twist your hand outward from your forehead." },
            { word: "KNOW", icon: "\u{1F9E0}", modelLabel: "KNOW", video: "videos/know.mp4", desc: "Touch your fingers to your forehead." }
        ]
    }
];

const matchDescriptions = {};
const matchMedia = {};
const matchDisplayNames = {};

function registerWordLookup(key, word) {
    if (!key) {
        return;
    }
    const normalized = normalizeLookupKey(key);
    matchDescriptions[normalized] = word.desc;
    matchMedia[normalized] = word.video;
    matchDisplayNames[normalized] = word.word;
}

for (const category of signCategories) {
    for (const word of category.words) {
        registerWordLookup(word.word, word);
        registerWordLookup(word.modelLabel, word);
        (word.aliases || []).forEach(alias => registerWordLookup(alias, word));
    }
}

const video = document.getElementById("webcam");
const outputCanvas = document.getElementById("output-canvas");
const canvasCtx = outputCanvas.getContext("2d");
const targetDisplay = document.getElementById("target-sign-text");
const targetDescription = document.getElementById("target-sign-description");
const dictionaryContainer = document.getElementById("dictionary-container");
const alphabetGrid = document.getElementById("alphabet-grid");
const progressBar = document.getElementById("accuracy-bar");
const progressText = document.getElementById("accuracy-text");
const progressStatus = document.getElementById("accuracy-status");
const successStars = document.getElementById("success-stars");

let datasetState = null;
let datasetPromise = null;
let handsModel = null;
let activeStream = null;
let animationFrameId = 0;
let currentMode = "practice";
let currentWordIndex = 0;
let currentAlphabetIndex = 0;
let holdStartedAt = 0;
let isAdvancing = false;
let latestPrediction = null;
let lastCameraError = "";
let cameraPermissionState = "prompt";

function normalizeLookupKey(value) {
    return String(value || "").replace(/\s+/g, "").replace(/['-]/g, "").toUpperCase();
}

function getDisplayDescription(label) {
    return matchDescriptions[normalizeLookupKey(label)] || "Show the sign to the camera.";
}

function getMediaPath(label) {
    return matchMedia[normalizeLookupKey(label)] || "";
}

function getDisplayName(label) {
    return matchDisplayNames[normalizeLookupKey(label)] || prettifyLabel(String(label || "").toUpperCase());
}

function isLetterLabel(label) {
    return /^[A-Z]$/.test(String(label || "").trim());
}

function sortSampleLandmarks(landmarks) {
    const sorted = Array.from(landmarks || []).sort((left, right) => (left.id || 0) - (right.id || 0));
    return sorted.length >= 21 ? sorted.slice(0, 21) : null;
}

function normalizeLandmarks(landmarks) {
    if (!Array.isArray(landmarks) || landmarks.length < 21) {
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
        const candidateHands = (sample.hands || []).filter(hand =>
            Array.isArray(hand.image_landmarks) && hand.image_landmarks.length >= 21
        );

        if (!candidateHands.length) {
            continue;
        }

        const primaryHand = candidateHands.reduce((best, hand) => {
            const score = hand.score || 0;
            return !best || score > (best.score || 0) ? hand : best;
        }, null);

        const normalized = normalizeLandmarks(sortSampleLandmarks(primaryHand.image_landmarks));
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

    const rawLabels = [...labelCounts.keys()];
    const alphabetLabels = rawLabels.filter(isLetterLabel).sort((left, right) => left.localeCompare(right));
    const otherWordLabels = rawLabels
        .filter(label => !isLetterLabel(label) && !PREFERRED_WORD_ORDER.includes(label))
        .sort((left, right) => left.localeCompare(right));
    const wordLabels = [
        ...PREFERRED_WORD_ORDER.filter(label => labelCounts.has(label)),
        ...otherWordLabels
    ];

    return {
        sampleCount: samples.length,
        labelCount: labelCounts.size,
        labelCounts,
        samples,
        wordLabels,
        alphabetLabels
    };
}

async function loadDataset() {
    const response = await fetch(DATASET_URL);
    if (!response.ok) {
        throw new Error(`Could not load landmarks_dataset.json (${response.status}).`);
    }

    const dataset = await response.json();
    const state = buildDatasetState(dataset);
    if (!state.sampleCount) {
        throw new Error("The JSON landmark dataset did not contain any usable samples.");
    }

    datasetState = state;
    return state;
}

function ensureDatasetLoaded() {
    if (!datasetPromise) {
        datasetPromise = loadDataset();
    }
    return datasetPromise;
}

function getActiveLabels() {
    if (!datasetState) {
        return [];
    }
    return currentMode === "alphabet" ? datasetState.alphabetLabels : datasetState.wordLabels;
}

function currentTargetLabel() {
    const labels = getActiveLabels();
    if (!labels.length) {
        return "";
    }
    return currentMode === "alphabet" ? labels[currentAlphabetIndex] : labels[currentWordIndex];
}

function updateCameraBadge(text, tone = "var(--accent-success)") {
    const badge = document.getElementById("camera-status-tag");
    if (!badge) {
        return;
    }
    badge.textContent = text;
    badge.style.background = tone;
}

function resetProgressUI() {
    const bar = document.getElementById("accuracy-bar");
    const text = document.getElementById("accuracy-text");
    const stars = document.getElementById("success-stars");
    const status = document.getElementById("accuracy-status");

    if (bar) {
        bar.style.height = "0%";
    }
    if (text) {
        text.innerText = "0";
    }
    if (stars) {
        stars.classList.remove("show");
    }
    if (status) {
        status.innerText = "Find the Sign";
    }
}

function updateTargetUI() {
    resetProgressUI();
    const target = currentTargetLabel();

    if (!target) {
        if (targetDisplay) {
            targetDisplay.innerText = "Waiting";
        }
        if (targetDescription) {
            targetDescription.innerText = datasetState
                ? currentMode === "alphabet"
                    ? "No letter samples are available in landmarks_dataset.json."
                    : "No word samples are available in landmarks_dataset.json."
                : "Loading JSON landmark matcher...";
        }
        return;
    }

    if (targetDisplay) {
        targetDisplay.innerText = getDisplayName(target);
    }
    if (targetDescription) {
        targetDescription.innerText = currentMode === "alphabet"
            ? `Show the letter "${target}" clearly in front of your camera.`
            : getDisplayDescription(target);
    }
}

function renderDictionary() {
    if (!dictionaryContainer) {
        return;
    }

    dictionaryContainer.innerHTML = signCategories.map(category => `
      <section class="word-category-section">
        <div class="category-header">
          <span class="category-icon">${category.icon}</span>
          <h2>${category.name}</h2>
        </div>
        <div class="word-grid">
          ${category.words.map(word => `
            <div class="word-card" onclick="openLearnWindow('${word.modelLabel}')">
              <div class="word-card-icon">${word.icon}</div>
              <h3>${word.word}</h3>
            </div>
          `).join("")}
        </div>
      </section>
    `).join("");
}

function renderAlphabet() {
    if (!alphabetGrid) {
        return;
    }

    if (!datasetState) {
        alphabetGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; padding: 2rem;">Loading JSON landmark dataset...</p>`;
        return;
    }

    if (!datasetState.alphabetLabels.length) {
        alphabetGrid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; padding: 2rem;">No alphabet labels are available in landmarks_dataset.json yet.</p>`;
        return;
    }

    alphabetGrid.innerHTML = datasetState.alphabetLabels.map(letter => `
      <div class="word-card" onclick="startLetterPractice('${letter}')">
        <div class="word-card-icon" style="font-size: 2.5rem; font-weight: 800; color: var(--accent-primary);">${letter}</div>
        <h3>Letter ${letter}</h3>
      </div>
    `).join("");
}

async function assetExists(path) {
    if (!path) {
        return false;
    }

    if (/^videos\//i.test(path)) {
        return AVAILABLE_MEDIA_ASSETS.has(path);
    }

    try {
        const response = await fetch(path, { method: "HEAD" });
        return response.ok;
    } catch (_error) {
        return false;
    }
}

function renderMediaFallback(container, label, description) {
    container.innerHTML = `
      <div class="media-fallback">
        <strong>${getDisplayName(label)}</strong>
        <p>${description}</p>
        <span>Demo media is not available in this project yet.</span>
      </div>
    `;
}

window.openLearnWindow = async (label) => {
    const modal = document.getElementById("modal-learn");
    const title = document.getElementById("modal-word-title");
    const mediaContainer = document.getElementById("modal-media-player");
    const description = document.getElementById("modal-description");

    const friendlyTitle = getDisplayName(label);
    const videoPath = getMediaPath(label);
    const text = getDisplayDescription(label);

    title.innerText = friendlyTitle;
    description.innerText = text;
    mediaContainer.innerHTML = `<div class="media-fallback"><strong>Loading...</strong></div>`;
    modal.classList.add("active");

    if (!(await assetExists(videoPath))) {
        renderMediaFallback(mediaContainer, label, text);
        return;
    }

    const isImage = /\.(gif|jpg|jpeg|png)$/i.test(videoPath);
    if (isImage) {
        mediaContainer.innerHTML = `<img src="${videoPath}" style="width:100%;height:100%;object-fit:cover;" alt="${friendlyTitle}">`;
        return;
    }

    mediaContainer.innerHTML = `
      <video autoplay loop muted playsinline style="width:100%;height:100%;object-fit:cover;">
        <source src="${videoPath}" type="video/mp4">
      </video>
    `;
};

window.closeLearnWindow = () => {
    document.getElementById("modal-learn").classList.remove("active");
    document.getElementById("modal-media-player").innerHTML = "";
};

window.speakTarget = () => {
    const text = document.getElementById("target-sign-text")?.innerText || "";
    if (!text) {
        return;
    }
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
};

window.skipSign = () => skipSign();
window.retryCamera = () => {
    stopCameraLoop();
    startCameraReal(currentMode).catch(error => {
        console.error("Retry camera failed", error);
    });
};

window.showPage = (pageId, clickedBtn) => {
    document.querySelectorAll(".page-content").forEach(page => page.classList.remove("active-page"));
    document.querySelectorAll(".nav-btn").forEach(button => button.classList.remove("active"));

    const page = document.getElementById(pageId);
    if (page) {
        page.classList.add("active-page");
    }
    if (clickedBtn) {
        clickedBtn.classList.add("active");
    }

    stopCameraLoop();
    if (pageId === "letter-page") {
        startCameraReal(currentMode).catch(error => {
            console.error("Practice start failed", error);
        });
    }
};

window.openPracticeMode = async (mode = "practice", clickedBtn = document.querySelectorAll(".nav-btn")[3]) => {
    currentMode = mode;
    await ensureDatasetLoaded();
    updateTargetUI();
    window.showPage("letter-page", clickedBtn);
};

window.startLetterPractice = async (char) => {
    await ensureDatasetLoaded();
    const nextIndex = datasetState?.alphabetLabels?.indexOf(char) ?? -1;
    if (nextIndex === -1) {
        return;
    }
    currentMode = "alphabet";
    currentAlphabetIndex = nextIndex;
    updateTargetUI();
    window.showPage("letter-page", document.querySelectorAll(".nav-btn")[3]);
};

const themeBtn = document.getElementById("theme-toggle");
if (themeBtn) {
    themeBtn.addEventListener("click", () => {
        const currentTheme = document.body.getAttribute("data-theme");
        const newTheme = currentTheme === "accessible" ? "default" : "accessible";
        document.body.setAttribute("data-theme", newTheme);

        const toggleText = themeBtn.querySelector(".toggle-text");
        const toggleIcon = themeBtn.querySelector(".toggle-icon");
        if (toggleText) {
            toggleText.innerText = newTheme === "accessible" ? "Standard Mode" : "Colorblind Mode";
        }
        if (toggleIcon) {
            toggleIcon.innerText = newTheme === "accessible" ? "Sun" : "View";
        }
    });
}

function skipSign() {
    const labels = getActiveLabels();
    if (!labels.length) {
        return;
    }

    if (currentMode === "alphabet") {
        currentAlphabetIndex = (currentAlphabetIndex + 1) % labels.length;
    } else {
        currentWordIndex = (currentWordIndex + 1) % labels.length;
    }

    holdStartedAt = 0;
    latestPrediction = null;
    updateTargetUI();
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
        return { primary: null };
    }

    const primary = liveHands.reduce((best, hand) =>
        !best || hand.handednessScore > best.handednessScore ? hand : best
    , null);

    return { primary };
}

function classifyCurrentHand(results) {
    const { primary } = getPrimaryLiveHand(results);
    if (!primary) {
        return { primaryHand: null, prediction: null };
    }

    const normalized = normalizeLandmarks(primary.landmarks);
    if (!normalized) {
        return { primaryHand: primary.landmarks, prediction: null };
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
            distance
        };
    }).sort((left, right) => left.distance - right.distance);

    if (!distances.length) {
        return { primaryHand: primary.landmarks, prediction: null };
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
        primaryHand: primary.landmarks,
        prediction: {
            predictedLabel,
            confidence: totalWeight ? predictedWeight / totalWeight : 0,
            similarity: 1 / (1 + bestDistance),
            bestDistance
        }
    };
}

function drawHand(rawHand) {
    outputCanvas.width = video.videoWidth || 1280;
    outputCanvas.height = video.videoHeight || 720;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(video, 0, 0, outputCanvas.width, outputCanvas.height);

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
            canvasCtx.fillStyle = index === 0
                ? "rgba(251, 191, 36, 0.98)"
                : DRAW_POINT_INDICES.includes(index)
                    ? "rgba(56, 189, 248, 0.98)"
                    : "rgba(30, 255, 30, 0.82)";
            canvasCtx.arc(
                landmark.x * outputCanvas.width,
                landmark.y * outputCanvas.height,
                index === 0 ? 6 : 4,
                0,
                Math.PI * 2
            );
            canvasCtx.fill();
        });
    }

    canvasCtx.restore();
}

function renderRecognition(recognition) {
    const target = currentTargetLabel();

    if (!recognition || !target) {
        if (progressStatus) {
            progressStatus.innerText = target
                ? "Show your hand to start."
                : datasetState
                    ? currentMode === "alphabet"
                        ? "No JSON letter tasks yet."
                        : "No JSON word tasks yet."
                    : "Loading JSON landmark matcher...";
        }
        if (progressBar) {
            progressBar.style.height = "0%";
        }
        if (progressText) {
            progressText.innerText = "0";
        }
        if (successStars) {
            successStars.classList.remove("show");
        }
        return;
    }

    const matchesTarget = normalizeLookupKey(recognition.predictedLabel) === normalizeLookupKey(target);
    const targetName = getDisplayName(target);
    const guessName = getDisplayName(recognition.predictedLabel);

    let visibleScore = matchesTarget ? Math.round(recognition.confidence * 100) : 0;

    if (matchesTarget && recognition.confidence >= MATCH_THRESHOLD) {
        if (!holdStartedAt) {
            holdStartedAt = performance.now();
        }

        const elapsed = (performance.now() - holdStartedAt) / 1000;
        visibleScore = Math.max(visibleScore, Math.min(100, Math.round((elapsed / HOLD_SECONDS) * 100)));

        if (progressStatus) {
            progressStatus.innerText = elapsed >= HOLD_SECONDS
                ? `Correct: ${targetName}`
                : `Good ${targetName}. Keep holding.`;
        }

        if (elapsed >= HOLD_SECONDS && !isAdvancing) {
            isAdvancing = true;
            if (successStars) {
                successStars.classList.add("show");
            }
            setTimeout(() => {
                skipSign();
                isAdvancing = false;
            }, 1000);
        }
    } else {
        holdStartedAt = 0;
        if (progressStatus) {
            progressStatus.innerText = matchesTarget
                ? `Looks like ${targetName}. Hold a little steadier.`
                : `I see ${guessName}. Show ${targetName}.`;
        }
        if (successStars) {
            successStars.classList.remove("show");
        }
    }

    if (progressBar) {
        progressBar.style.height = `${visibleScore}%`;
    }
    if (progressText) {
        progressText.innerText = String(visibleScore);
    }
}

async function initHands() {
    if (handsModel || typeof window.Hands === "undefined") {
        return;
    }

    handsModel = new window.Hands({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    handsModel.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
    });
    handsModel.onResults(results => {
        if (!activeStream) {
            return;
        }
        const { primaryHand, prediction } = datasetState
            ? classifyCurrentHand(results)
            : { primaryHand: null, prediction: null };
        latestPrediction = prediction;
        drawHand(primaryHand);
        renderRecognition(prediction);
    });
}

async function startCameraReal(mode) {
    currentMode = mode || currentMode || "practice";
    resetProgressUI();
    holdStartedAt = 0;
    isAdvancing = false;
    latestPrediction = null;
    lastCameraError = "";
    updateTargetUI();
    updateCameraBadge("Loading", "var(--accent-secondary)");

    try {
        await ensureDatasetLoaded();
        await initHands();

        if (!handsModel) {
            throw new Error("MediaPipe Hands did not load on this page.");
        }

        cameraPermissionState = await getCameraPermissionState();

        if (!activeStream) {
            activeStream = await startCameraStream();
            video.srcObject = activeStream;
            await video.play();
        }

        updateCameraBadge("Active", "var(--accent-success)");

        const processFrame = async () => {
            if (!activeStream) {
                return;
            }
            if (video.readyState >= 2 && handsModel) {
                await handsModel.send({ image: video });
            }
            if (activeStream) {
                animationFrameId = requestAnimationFrame(processFrame);
            }
        };
        animationFrameId = requestAnimationFrame(processFrame);
    } catch (error) {
        lastCameraError = describeCameraError(error);
        console.error("Camera or JSON matcher failed", error);
        updateCameraBadge("Error", "var(--accent-danger)");
        if (progressStatus) {
            progressStatus.innerText = lastCameraError;
        }
    }
}

function stopCameraLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = 0;
    }

    stopMediaStream(activeStream);
    activeStream = null;
    video.srcObject = null;
    updateCameraBadge("Idle", "var(--text-heading)");
}

window.addEventListener("beforeunload", () => stopCameraLoop());

window.addEventListener("DOMContentLoaded", async () => {
    renderDictionary();
    renderAlphabet();
    updateTargetUI();

    try {
        await ensureDatasetLoaded();
        renderAlphabet();
        updateTargetUI();
    } catch (error) {
        console.error("Dataset load failed", error);
        if (progressStatus) {
            progressStatus.innerText = error.message;
        }
    }
});
