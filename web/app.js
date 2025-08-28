const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const labelEl = document.getElementById('label');
const aadharEl = document.getElementById('aadhar');
const nameEl = document.getElementById('name');
const framesTotalEl = document.getElementById('framesTotal');
const everyNEl = document.getElementById('everyN');
const regStatusEl = document.getElementById('reg-status');
const partyEl = document.getElementById('party');
const voteStatusEl = document.getElementById('vote-status');

let lastLabel = null;

async function init() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
  } catch (e) {
    alert('Could not access webcam: ' + e.message);
  }
  try {
    const cfg = await (await fetch('/api/config')).json();
    if (cfg?.parties) {
      partyEl.innerHTML = cfg.parties.map(p => `<option value="${p}">${p}</option>`).join('');
    }
  } catch (e) {}
}

function snapshotDataUrl() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.9);
}

document.getElementById('btn-capture').addEventListener('click', () => {
  snapshotDataUrl();
});

document.getElementById('btn-predict').addEventListener('click', async () => {
  const dataUrl = snapshotDataUrl();
  labelEl.textContent = '...';
  try {
    const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: dataUrl }) });
    const js = await res.json();
    if (!res.ok) throw new Error(js.error || 'Prediction failed');
    lastLabel = js.label;
    labelEl.textContent = js.label;
  } catch (e) {
    labelEl.textContent = '-';
    alert(e.message);
  }
});

document.getElementById('btn-register').addEventListener('click', async () => {
  const aadhar = aadharEl.value.trim();
  const framesTotal = Math.max(1, Math.min(5, parseInt(framesTotalEl.value || '5', 10)));
  const everyN = Math.max(1, Math.min(10, parseInt(everyNEl.value || '2', 10)));
  if (!aadhar) return alert('Enter aadhar number first');

  regStatusEl.textContent = 'Collecting images...';
  const images = [];
  let collected = 0, i = 0;
  while (collected < framesTotal) {
    if (i % everyN === 0) {
      images.push(snapshotDataUrl());
      collected++;
    }
    i++;
    await new Promise(r => setTimeout(r, 80));
  }
  regStatusEl.textContent = 'Uploading...';
  try {
    const name = nameEl.value.trim();
    const res = await fetch('/api/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ aadhar, name, images }) });
    const js = await res.json();
    if (!res.ok) {
      if (js?.error === 'duplicate_face') {
        regStatusEl.textContent = `This face seems registered to ${js.registered_to}.`;
        alert('Duplicate: already registered to ' + js.registered_to);
        return;
      }
      throw new Error(js.error || 'Registration failed');
    }
    regStatusEl.textContent = `Registered ${js.added} frames for ${aadhar}`;
  } catch (e) {
    regStatusEl.textContent = e.message;
    alert(e.message);
  }
});

document.getElementById('btn-vote').addEventListener('click', async () => {
  const party = partyEl.value;
  const voter = lastLabel || aadharEl.value.trim();
  if (!voter) return alert('Predict or enter aadhar to vote');
  voteStatusEl.textContent = 'Submitting vote...';
  try {
    const res = await fetch('/api/vote', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ voter, party }) });
    const js = await res.json();
    if (!res.ok) throw new Error(js.error || 'Vote failed');
    voteStatusEl.textContent = 'Vote recorded!';
  } catch (e) {
    voteStatusEl.textContent = e.message;
    alert(e.message);
  }
});

init();



