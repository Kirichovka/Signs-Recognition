export const MAX_SEQUENCE = 40;
export const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];
const DEFAULT_MODEL_NAME = "everyday_daily_v1";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let cachedModel = null;

function resolveRequestedModelName() {
    const params = new URLSearchParams(window.location.search);
    const requested = (params.get("model") || "").trim();
    return requested || DEFAULT_MODEL_NAME;
}

function buildModelUrls(modelName) {
    return {
        modelUrl: new URL(`../models/${modelName}.onnx`, import.meta.url),
        metadataUrl: new URL(`../models/${modelName}_metadata.json`, import.meta.url)
    };
}

export function prettifyLabel(label) {
    return label
        .replace(/[0-9]+$/g, "")
        .split(/[_/]/)
        .filter(Boolean)
        .map(chunk => chunk.charAt(0) + chunk.slice(1).toLowerCase())
        .join(" / ");
}

export function softmax(values) {
    const maxValue = Math.max(...values);
    const exps = values.map(value => Math.exp(value - maxValue));
    const sum = exps.reduce((acc, value) => acc + value, 0);
    return exps.map(value => value / sum);
}

export async function loadBrowserModel() {
    if (cachedModel) {
        return cachedModel;
    }

    const modelName = resolveRequestedModelName();
    const { modelUrl, metadataUrl } = buildModelUrls(modelName);

    const metadataResponse = await fetch(metadataUrl);
    if (!metadataResponse.ok) {
        throw new Error(`Could not load metadata for ${modelName} (${metadataResponse.status}).`);
    }
    const metadata = await metadataResponse.json();

    if (!globalThis.ort) {
        throw new Error("onnxruntime-web did not load on this page.");
    }

    globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;
    const session = await globalThis.ort.InferenceSession.create(modelUrl.href, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
    });

    cachedModel = {
        ...metadata,
        requested_model_name: modelName,
        session,
        inputName: session.inputNames[0],
        outputName: session.outputNames[0]
    };
    return cachedModel;
}

export async function predictWithBrowserModel(model, sequence) {
    if (sequence.length !== model.sequence_length) {
        throw new Error(`Expected ${model.sequence_length} frames, got ${sequence.length}.`);
    }

    const flattened = new Float32Array(model.sequence_length * model.feature_size);
    for (let frameIndex = 0; frameIndex < sequence.length; frameIndex += 1) {
        const frame = sequence[frameIndex];
        if (frame.length !== model.feature_size) {
            throw new Error(`Expected frame width ${model.feature_size}, got ${frame.length}.`);
        }
        flattened.set(frame, frameIndex * model.feature_size);
    }

    const tensor = new globalThis.ort.Tensor("float32", flattened, [1, model.sequence_length, model.feature_size]);
    const outputs = await model.session.run({ [model.inputName]: tensor });
    const logitsTensor = outputs[model.outputName];
    const logits = Array.from(logitsTensor.data);
    const probabilities = softmax(logits);

    const predictions = probabilities
        .map((score, index) => ({ label: model.label_names[index], score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, model.top_k || 5);

    return {
        predictions,
        logits,
        sequence_length: model.sequence_length,
        feature_size: model.feature_size
    };
}

export function emptyPose() {
    return Array.from({ length: POSE_IDS.length }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
}

export function toLandmarkArray(landmarks, count) {
    if (!landmarks?.length) {
        return Array.from({ length: count }, () => ({ x: 0, y: 0, z: 0 }));
    }
    return Array.from({ length: count }, (_, index) => landmarks[index] || { x: 0, y: 0, z: 0 });
}

export function toPoseSubset(landmarks) {
    if (!landmarks?.length) {
        return emptyPose();
    }
    return POSE_IDS.map(index => landmarks[index] || { x: 0, y: 0, z: 0, visibility: 0 });
}

export function normalizeLandmarks(leftHand, rightHand, pose) {
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

export function featureVectorFromResults(results) {
    const leftHand = toLandmarkArray(results.leftHandLandmarks, 21);
    const rightHand = toLandmarkArray(results.rightHandLandmarks, 21);
    const pose = toPoseSubset(results.poseLandmarks);
    return normalizeLandmarks(leftHand, rightHand, pose);
}

export function drawHolisticResults(canvasCtx, outputCanvas, inputVideo, results) {
    outputCanvas.width = inputVideo.videoWidth || 1280;
    outputCanvas.height = inputVideo.videoHeight || 720;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    canvasCtx.translate(outputCanvas.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    const drawLandmarkSet = (landmarks, connections, connectorColor, fillColor) => {
        if (!landmarks) { return; }
        globalThis.drawConnectors(canvasCtx, landmarks, connections, { color: connectorColor, lineWidth: 3 });
        globalThis.drawLandmarks(canvasCtx, landmarks, { color: "#eff6ff", fillColor, radius: 3 });
    };

    drawLandmarkSet(results.poseLandmarks, globalThis.POSE_CONNECTIONS, "#fb7185", "#be123c");
    drawLandmarkSet(results.leftHandLandmarks, globalThis.HAND_CONNECTIONS, "#f97316", "#7c2d12");
    drawLandmarkSet(results.rightHandLandmarks, globalThis.HAND_CONNECTIONS, "#38bdf8", "#1d4ed8");
    canvasCtx.restore();
}

export function stopMediaStream(stream) {
    if (!stream) {
        return;
    }
    stream.getTracks().forEach(track => track.stop());
}

export async function getCameraPermissionState() {
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

export async function enumerateVideoDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) {
        return [];
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(device => device.kind === "videoinput");
}

export async function startCameraStream() {
    let videoDevices = [];
    try {
        videoDevices = await enumerateVideoDevices();
    } catch (error) {
        console.warn("Could not enumerate video devices before camera start.", error);
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

    let lastError = null;
    for (const constraints of attempts) {
        try {
            return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
            lastError = error;
            if (error?.name !== "NotFoundError" && error?.name !== "OverconstrainedError") {
                throw error;
            }
        }
    }
    throw lastError || new Error("Camera access failed.");
}

export function describeCameraError(error) {
    if (!error) { return "Camera access failed."; }
    if (error.name === "NotAllowedError") { return "Camera permission was blocked. Allow access in the browser and try again."; }
    if (error.name === "NotFoundError") { return "No camera was found on this device."; }
    if (error.name === "NotReadableError") { return "The camera is busy in another app. Close Zoom, Teams, OBS, or the Windows Camera app, then try again."; }
    if (error.name === "OverconstrainedError") { return "The requested camera settings are not supported on this device."; }
    return `Camera error: ${error.message || error.name}`;
}
