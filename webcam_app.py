from collections import deque

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

from transformer import ActionClassificationTransformer, LABELS, N_STEPS


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ActionSnap Webcam</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101413;
      color: #f3f5ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      background: #101413;
    }
    main {
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .stage {
      width: min(100%, 980px);
      aspect-ratio: 16 / 10;
      position: relative;
      overflow: hidden;
      border: 1px solid #303a36;
      background: #050706;
      border-radius: 8px;
    }
    video, canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    video.mirror, canvas.mirror {
      transform: scaleX(-1);
    }
    aside {
      border-left: 1px solid #303a36;
      padding: 28px 24px;
      background: #171d1b;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }
    .label {
      font-size: clamp(34px, 8vw, 58px);
      line-height: 1;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .muted { color: #9ba7a1; font-size: 14px; line-height: 1.5; }
    .meter {
      height: 10px;
      border-radius: 999px;
      background: #2a3430;
      overflow: hidden;
    }
    .bar {
      height: 100%;
      width: 0%;
      background: #82d173;
      transition: width 140ms ease;
    }
    button {
      appearance: none;
      border: 1px solid #46524d;
      background: #e6f2df;
      color: #111713;
      border-radius: 8px;
      min-height: 42px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #202824;
      color: #f3f5ef;
    }
    input[type="file"] {
      width: 100%;
      color: #c7d0cb;
      font-size: 13px;
    }
    .rows {
      display: grid;
      gap: 10px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      font-size: 13px;
      color: #c7d0cb;
    }
    @media (max-width: 860px) {
      body { grid-template-columns: 1fr; }
      aside { border-left: 0; border-top: 1px solid #303a36; }
      main { min-height: auto; }
    }
  </style>
</head>
<body>
  <main>
    <div class="stage">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="overlay"></canvas>
    </div>
  </main>
  <aside>
    <h1>ActionSnap Webcam</h1>
    <div>
      <div id="label" class="label">Ready</div>
      <p id="status" class="muted">Click Start and allow camera access. Prediction begins after 32 detected pose frames.</p>
    </div>
    <div>
      <div class="row"><span>Confidence</span><span id="conf">0%</span></div>
      <div class="meter"><div id="bar" class="bar"></div></div>
    </div>
    <div class="rows" id="scores"></div>
    <button id="start">Start Webcam</button>
    <input id="upload" type="file" accept="video/*">
    <button id="stop" class="secondary">Stop</button>
    <p class="muted">This uses MediaPipe Pose Landmarker in the browser, then sends compact 18-keypoint frames to the local Flask model server.</p>
  </aside>
<script type="module">
import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const label = document.getElementById("label");
const statusEl = document.getElementById("status");
const conf = document.getElementById("conf");
const bar = document.getElementById("bar");
const scoresEl = document.getElementById("scores");
const upload = document.getElementById("upload");
let stream = null;
let timer = null;
let landmarker = null;
let lastVideoTime = -1;
let objectUrl = null;

function fitCanvas() {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
}

function point(landmarks, index) {
  const lm = landmarks[index];
  if (!lm || (lm.visibility ?? 1) < 0.35) return [0, 0];
  return [lm.x * 640, lm.y * 480];
}

function midpoint(a, b) {
  if ((a[0] === 0 && a[1] === 0) || (b[0] === 0 && b[1] === 0)) return [0, 0];
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

function toOpenPose18(landmarks) {
  const nose = point(landmarks, 0);
  const rShoulder = point(landmarks, 12);
  const rElbow = point(landmarks, 14);
  const rWrist = point(landmarks, 16);
  const lShoulder = point(landmarks, 11);
  const lElbow = point(landmarks, 13);
  const lWrist = point(landmarks, 15);
  const rHip = point(landmarks, 24);
  const rKnee = point(landmarks, 26);
  const rAnkle = point(landmarks, 28);
  const lHip = point(landmarks, 23);
  const lKnee = point(landmarks, 25);
  const lAnkle = point(landmarks, 27);
  const rEye = point(landmarks, 5);
  const lEye = point(landmarks, 2);
  const rEar = point(landmarks, 8);
  const lEar = point(landmarks, 7);
  const neck = midpoint(lShoulder, rShoulder);
  return [
    nose, neck, rShoulder, rElbow, rWrist, lShoulder, lElbow, lWrist,
    rHip, rKnee, rAnkle, lHip, lKnee, lAnkle, rEye, lEye, rEar, lEar
  ].flat();
}

async function ensureLandmarker() {
  if (landmarker) return;
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
}

async function resetPrediction() {
  await fetch("/api/reset", {method: "POST"});
  lastVideoTime = -1;
  label.textContent = "Ready";
  conf.textContent = "0%";
  bar.style.width = "0%";
  scoresEl.innerHTML = "";
}

async function sendFeatures(features) {
  const res = await fetch("/api/predict_features", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({features})
  });
  const data = await res.json();
  statusEl.textContent = data.status;
  if (data.label) {
    label.textContent = data.label;
    const pct = Math.round(data.confidence * 100);
    conf.textContent = `${pct}%`;
    bar.style.width = `${pct}%`;
    scoresEl.innerHTML = data.scores.map(([name, value]) => (
      `<div class="row"><span>${name}</span><span>${Math.round(value * 100)}%</span></div>`
    )).join("");
  }
}

async function predict() {
  if (video.readyState < 2 || !landmarker) return;
  fitCanvas();
  ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const result = landmarker.detectForVideo(video, performance.now());
  if (!result.landmarks || !result.landmarks.length) {
    statusEl.textContent = "No pose detected. Step back and keep your full body visible.";
    return;
  }

  const landmarks = result.landmarks[0];
  const features = toOpenPose18(landmarks);
  try {
    await sendFeatures(features);
  } catch (err) {
    statusEl.textContent = `Prediction error: ${err}`;
  }
}

function stopCurrent() {
  clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
}

document.getElementById("start").onclick = async () => {
  stopCurrent();
  statusEl.textContent = "Loading pose detector...";
  await ensureLandmarker();
  await resetPrediction();
  video.classList.add("mirror");
  overlay.classList.add("mirror");
  video.controls = false;
  stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
  video.srcObject = stream;
  statusEl.textContent = "Camera running. Collecting pose frames...";
  timer = setInterval(predict, 180);
};

upload.onchange = async () => {
  const file = upload.files && upload.files[0];
  if (!file) return;
  stopCurrent();
  statusEl.textContent = "Loading pose detector...";
  await ensureLandmarker();
  await resetPrediction();
  video.classList.remove("mirror");
  overlay.classList.remove("mirror");
  video.controls = true;
  video.muted = true;
  objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;
  await video.play();
  statusEl.textContent = `Analyzing ${file.name}. Collecting pose frames...`;
  timer = setInterval(() => {
    if (video.ended) {
      clearInterval(timer);
      timer = null;
      statusEl.textContent = "Video analysis finished.";
      return;
    }
    predict();
  }, 180);
};

document.getElementById("stop").onclick = () => {
  stopCurrent();
  statusEl.textContent = "Stopped.";
};
</script>
</body>
</html>
"""


app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ActionClassificationTransformer.load_from_checkpoint("models/saved_model.ckpt")
model.to(device)
model.eval()
sequence = deque(maxlen=N_STEPS)


@app.get("/")
def index():
    return render_template_string(HTML)


@app.post("/api/reset")
def reset():
    sequence.clear()
    return jsonify(status="reset")


@app.post("/api/predict_features")
def predict_features():
    payload = request.get_json(force=True)
    features = np.array(payload["features"], dtype=np.float32)

    if features.shape != (36,):
        return jsonify(status="Expected 36 features from 18 pose keypoints.", label=None), 400

    sequence.append(features)
    if len(sequence) < N_STEPS:
        return jsonify(status=f"Collecting pose frames: {len(sequence)}/{N_STEPS}", label=None)

    seq = np.array(sequence)
    x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).detach().cpu().numpy()

    best = int(probs.argmax())
    scores = sorted(zip(LABELS, probs.tolist()), key=lambda item: item[1], reverse=True)
    return jsonify(
        status="Live prediction running.",
        label=LABELS[best],
        confidence=float(probs[best]),
        scores=scores,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
