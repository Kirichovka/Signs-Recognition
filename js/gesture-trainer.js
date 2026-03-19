import {
    ALPHABET_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    MAX_SEQUENCE,
    drawHolisticResults,
    featureVectorFromResults,
    loadBrowserModel,
    predictWithBrowserModel,
    prettifyLabel,
    startCameraStream
} from "./sign-model-runtime.js";

const signCategories = [
    {
        name: "Greetings & Basics",
        icon: "\u{1F44B}",
        words: [
            { word: "HELLO", icon: "\u{1F44B}", modelLabel: "HELLO", video: "videos/hello.mp4", desc: "Raise your dominant hand near your forehead and move it outward." },
            { word: "BYE", icon: "\u{1F44B}", modelLabel: "BYE", video: "videos/bye.mp4", desc: "Open your hand and wave side to side." },
            { word: "PLEASE", icon: "\u{1F970}", modelLabel: "PLEASE", video: "videos/please.mp4", desc: "Place your flat hand on your chest and move it in a circular motion." },
            { word: "THANK YOU", icon: "\u{1F64F}", modelLabel: "THANKYOU", video: "videos/thankyou.mp4", desc: "Touch your fingers to your chin, then move your hand forward." },
            { word: "SORRY", icon: "\u{1F614}", modelLabel: "SORRY", video: "videos/sorry.mp4", desc: "Make a fist and rub it in a circular motion on your chest." },
            { word: "YES", icon: "\u{1F44D}", modelLabel: "YES", video: "videos/yes.mp4", desc: "Make a fist and nod it up and down." },
            { word: "NO", icon: "\u{1F44E}", modelLabel: "NO", video: "videos/no.mp4", desc: "Pinch your index and middle fingers to your thumb." },
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

const MODE_MODEL_NAMES = {
    practice: DEFAULT_MODEL_NAME,
    alphabet: ALPHABET_MODEL_NAME
};

const matchDescriptions = {};
const matchMedia = {};
const matchDisplayNames = {};
for (const category of signCategories) {
    for (const word of category.words) {
        matchDescriptions[word.word] = word.desc;
        matchDescriptions[word.modelLabel] = word.desc;
        matchMedia[word.word] = word.video;
        matchMedia[word.modelLabel] = word.video;
        matchDisplayNames[word.word] = word.word;
        matchDisplayNames[word.modelLabel] = word.word;
    }
}

const modelCache = new Map();
let holistic = null;
let activeStream = null;
let frameBuffer = [];
let isPredicting = false;
let currentTargetIndex = 0;
let currentAlphaIndex = 0;
let currentMode = "practice";
let holdStartedAt = 0;
let targetLabels = [];
let alphaLabels = [];
let alphabetModelReady = false;
let isAdvancing = false;

const HOLD_SECONDS = 0.8;
const MATCH_THRESHOLD = 0.45;

function normalizeLookupKey(value) {
    return String(value || "").replace(/\s+/g, "").replace(/['-]/g, "").toUpperCase();
}

function getDisplayDescription(label) {
    return matchDescriptions[label] || matchDescriptions[normalizeLookupKey(label)] || "Show the sign to the camera.";
}

function getMediaPath(label) {
    return matchMedia[label] || matchMedia[normalizeLookupKey(label)] || "";
}

function getDisplayName(label) {
    return matchDisplayNames[label] || matchDisplayNames[normalizeLookupKey(label)] || prettifyLabel(label);
}

async function ensureModel(mode) {
    if (modelCache.has(mode)) {
        return modelCache.get(mode);
    }

    const requestedName = MODE_MODEL_NAMES[mode] || DEFAULT_MODEL_NAME;
    const model = await loadBrowserModel(requestedName);
    modelCache.set(mode, model);

    if (mode === "practice") {
        targetLabels = model.label_names || [];
    } else if (mode === "alphabet") {
        alphaLabels = model.label_names || [];
        alphabetModelReady = true;
    }

    return model;
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
    const display = document.getElementById("target-sign-text");
    const description = document.getElementById("target-sign-description");

    resetProgressUI();

    if (currentMode === "alphabet") {
        const letter = alphaLabels[currentAlphaIndex] || "A";
        if (display) {
            display.innerText = letter;
        }
        if (description) {
            description.innerText = `Sign the letter "${letter}" clearly in front of your camera.`;
        }
        return;
    }

    if (!targetLabels.length) {
        if (display) {
            display.innerText = "...";
        }
        if (description) {
            description.innerText = "Loading the practice model...";
        }
        return;
    }

    const label = targetLabels[currentTargetIndex];
    if (display) {
        display.innerText = getDisplayName(label);
    }
    if (description) {
        description.innerText = getDisplayDescription(label);
    }
}

function renderDictionary() {
    const container = document.getElementById("dictionary-container");
    if (!container) {
        return;
    }

    container.innerHTML = signCategories.map(category => `
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
    const grid = document.getElementById("alphabet-grid");
    if (!grid) {
        return;
    }

    if (!alphabetModelReady || !alphaLabels.length) {
        grid.innerHTML = `<p style="grid-column: 1 / -1; text-align: center; padding: 2rem;">Loading alphabet model...</p>`;
        return;
    }

    grid.innerHTML = alphaLabels.map(letter => `
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
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    speechSynthesis.speak(utterance);
};

window.skipSign = () => skipSign();
window.retryCamera = () => {
    stopCameraReal();
    startCameraReal(currentMode);
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

    stopCameraReal();
    if (pageId === "letter-page") {
        startCameraReal(currentMode);
    }
};

window.openPracticeMode = (mode = "practice", clickedBtn = document.querySelectorAll(".nav-btn")[3]) => {
    currentMode = mode;
    window.showPage("letter-page", clickedBtn);
    updateTargetUI();
};

window.startLetterPractice = async (char) => {
    await ensureModel("alphabet");
    const nextIndex = alphaLabels.indexOf(char);
    if (nextIndex === -1) {
        return;
    }
    currentMode = "alphabet";
    currentAlphaIndex = nextIndex;
    window.showPage("letter-page", document.querySelectorAll(".nav-btn")[3]);
    updateTargetUI();
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
    if (currentMode === "alphabet") {
        if (!alphaLabels.length) {
            return;
        }
        currentAlphaIndex = (currentAlphaIndex + 1) % alphaLabels.length;
    } else if (targetLabels.length) {
        currentTargetIndex = (currentTargetIndex + 1) % targetLabels.length;
    }
    updateTargetUI();
}

async function initHolistic(video, canvas) {
    if (holistic || typeof window.Holistic === "undefined") {
        return;
    }

    const ctx = canvas.getContext("2d");
    holistic = new window.Holistic({
        locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
    holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    holistic.onResults(async results => {
        if (!activeStream) {
            return;
        }

        drawHolisticResults(ctx, canvas, video, results);

        if (currentMode === "practice") {
            frameBuffer.push(featureVectorFromResults(results));
            if (frameBuffer.length > MAX_SEQUENCE) {
                frameBuffer.shift();
            }
            if (frameBuffer.length === MAX_SEQUENCE && !isPredicting && !isAdvancing) {
                await runPrediction(video);
            }
            return;
        }

        if (!isPredicting && !isAdvancing) {
            await runPrediction(video);
        }
    });
}

async function startCameraReal(mode) {
    currentMode = mode || currentMode || "practice";
    const video = document.getElementById("webcam");
    const canvas = document.getElementById("output-canvas");
    if (!video || !canvas) {
        return;
    }

    resetProgressUI();
    frameBuffer = [];
    holdStartedAt = 0;
    isAdvancing = false;
    updateTargetUI();
    updateCameraBadge("Loading", "var(--accent-secondary)");

    try {
        await ensureModel(currentMode);
        await initHolistic(video, canvas);

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
            if (video.readyState >= 2 && holistic) {
                await holistic.send({ image: video });
            }
            if (activeStream) {
                requestAnimationFrame(processFrame);
            }
        };
        requestAnimationFrame(processFrame);
    } catch (error) {
        console.error("Camera or model failed", error);
        updateCameraBadge("Error", "var(--accent-danger)");
        const status = document.getElementById("accuracy-status");
        if (status) {
            status.innerText = "Camera or model failed";
        }
    }
}

async function runPrediction(video) {
    const model = modelCache.get(currentMode) || await ensureModel(currentMode);
    if (!model) {
        return;
    }

    isPredicting = true;
    try {
        const input = model.model_type === "image" ? video : frameBuffer;
        if (model.model_type !== "image" && frameBuffer.length < MAX_SEQUENCE) {
            return;
        }

        const result = await predictWithBrowserModel(model, input);
        const best = result.predictions[0];
        const progressBar = document.getElementById("accuracy-bar");
        const accuracyText = document.getElementById("accuracy-text");
        const successStars = document.getElementById("success-stars");
        const accuracyStatus = document.getElementById("accuracy-status");

        const targetLabel = currentMode === "alphabet"
            ? alphaLabels[currentAlphaIndex]
            : targetLabels[currentTargetIndex];
        const targetPrediction = result.predictions.find(prediction =>
            prediction.label.toLowerCase() === String(targetLabel).toLowerCase()
        );
        const score = targetPrediction ? targetPrediction.score : 0;
        let displayScore = Math.floor(score * 100);

        if (best && targetLabel && best.label.toLowerCase() === String(targetLabel).toLowerCase() && best.score > MATCH_THRESHOLD) {
            if (!holdStartedAt) {
                holdStartedAt = performance.now();
            }
            const holdTime = (performance.now() - holdStartedAt) / 1000;
            displayScore = Math.min(100, Math.floor((holdTime / HOLD_SECONDS) * 100));

            if (accuracyStatus) {
                accuracyStatus.innerText = "HOLD STEADY!";
            }

            if (holdTime >= HOLD_SECONDS) {
                if (successStars) {
                    successStars.classList.add("show");
                }
                isAdvancing = true;
                setTimeout(() => {
                    skipSign();
                    holdStartedAt = 0;
                    isAdvancing = false;
                }, 1200);
                return;
            }
        } else {
            holdStartedAt = 0;
            if (accuracyStatus) {
                accuracyStatus.innerText = score > 0.1 ? "Match Found!" : "Searching...";
            }
        }

        if (progressBar) {
            progressBar.style.height = `${displayScore}%`;
        }
        if (accuracyText) {
            accuracyText.innerText = String(displayScore);
        }
    } catch (error) {
        console.error("Prediction failed", error);
    } finally {
        isPredicting = false;
    }
}

function stopCameraReal() {
    if (activeStream) {
        activeStream.getTracks().forEach(track => track.stop());
        activeStream = null;
    }
    const video = document.getElementById("webcam");
    if (video) {
        video.srcObject = null;
    }
    updateCameraBadge("Idle", "var(--text-heading)");
}

window.addEventListener("DOMContentLoaded", async () => {
    renderDictionary();
    renderAlphabet();

    try {
        await ensureModel("practice");
        updateTargetUI();
    } catch (error) {
        console.error("Practice model load failed", error);
    }

    try {
        await ensureModel("alphabet");
        renderAlphabet();
    } catch (error) {
        console.error("Alphabet model load failed", error);
    }
});
