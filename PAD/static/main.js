const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const stopBtn  = document.getElementById("stop");
const snapBtn  = document.getElementById("snap");
const decision = document.getElementById("decision");
const prob = document.getElementById("prob");
const thr  = document.getElementById("thr");
const lat  = document.getElementById("lat");
const opSel = document.getElementById("op");
const loopChk = document.getElementById("loop");
const health = document.getElementById("health");

let stream = null;
let loopHandle = null;

async function checkHealth() {
  try {
    const r = await fetch("/healthz");
    const j = await r.json();
    health.textContent = `${j.status} (${j.img_size}px, ${j.device})`;
  } catch {
    health.textContent = "unavailable";
  }
}
checkHealth();

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
  } catch (e) {
    alert("Failed to access camera. Use Chrome/Edge on http://localhost and allow camera permissions.");
    console.error(e);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (loopHandle) {
    clearInterval(loopHandle);
    loopHandle = null;
  }
}

async function analyzeFrame() {
  if (!stream) return;
  // draw to 256x256 canvas
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const t0 = performance.now();
  const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.95));
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");

  const op = opSel.value; // "bpc5" or "minacer"
  const r = await fetch(`/predict?op=${op}`, { method: "POST", body: fd });
  const j = await r.json();
  const t1 = performance.now();

  prob.textContent = j.prob_attack.toFixed(4);
  thr.textContent  = j.threshold.toFixed(3);
  lat.textContent  = (t1 - t0).toFixed(1);

  decision.textContent = j.decision;
  decision.className = "pill " + (j.decision === "attack" ? "bad" : "ok");
}

startBtn.onclick = async () => {
  await startCamera();
  if (loopChk.checked && !loopHandle) {
    loopHandle = setInterval(analyzeFrame, 600); // every ~0.6s
  }
};
stopBtn.onclick = () => stopCamera();
snapBtn.onclick = () => analyzeFrame();
loopChk.onchange = () => {
  if (loopChk.checked && stream && !loopHandle) loopHandle = setInterval(analyzeFrame, 600);
  else if (!loopChk.checked && loopHandle) { clearInterval(loopHandle); loopHandle = null; }
};