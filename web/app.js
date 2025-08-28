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
const confirmIdEl = document.getElementById('confirm-id');
const confirmPartyEl = document.getElementById('confirm-party');
const confirmNameEl = document.getElementById('confirm-name');
const btnTheme = document.getElementById('btn-theme');
const btnFontInc = document.getElementById('font-inc');
const btnFontDec = document.getElementById('font-dec');
const btnKiosk = document.getElementById('btn-kiosk');
const toasts = document.getElementById('toasts');
const videoReg = document.getElementById('video-reg');

let lastLabel = null;
let inactivityTimer = null;
let capturedImages = [];

async function init() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    if (videoReg) videoReg.srcObject = stream;
  } catch (e) {
    alert('Could not access webcam: ' + e.message);
  }
  try {
    const cfg = await (await fetch('/api/config')).json();
    if (cfg?.parties) {
      partyEl.innerHTML = cfg.parties.map(p => `<option value="${p}">${p}</option>`).join('');
    }
  } catch (e) {}
  // If Aadhar typed manually, check registry quickly
  aadharEl?.addEventListener('change', async () => {
    const id = aadharEl.value.trim();
    if (!id) return;
    try {
      const reg = await (await fetch(`/api/registry/${encodeURIComponent(id)}`)).json();
      if (!reg.registry) toast('Aadhar not found in registry');
      if (reg.registryName && !nameEl.value) nameEl.value = reg.registryName;
    } catch (e) {}
  });
}

function snapshotDataUrl(src = video) {
  const w = (src || video).videoWidth || 640;
  const h = (src || video).videoHeight || 480;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(src || video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', 0.9);
}

document.getElementById('btn-predict').addEventListener('click', async () => {
  const dataUrl = snapshotDataUrl();
  labelEl.textContent = '...';
  try {
    const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: dataUrl }) });
    const js = await res.json();
    if (!res.ok) {
      if (js.error === 'no face detected' || js.error === 'no training data') {
        toast('Could not identify. You can register as a new user.');
        goToStep(2);
        return;
      }
      throw new Error(js.error || 'Prediction failed');
    }
    lastLabel = js.label;
    labelEl.textContent = js.label;
    aadharEl.value = js.label;
    // Try to fetch profile to auto-fill name
    try {
      const prof = await (await fetch(`/api/profile/${encodeURIComponent(js.label)}`)).json();
      if (prof?.profile?.name) nameEl.value = prof.profile.name;
    } catch (e) {}
    goToStep(3);
    toast('Identified as ' + js.label);
  } catch (e) {
    labelEl.textContent = '-';
    toast(e.message);
  }
});

document.getElementById('btn-capture-faces')?.addEventListener('click', async () => {
  const framesTotal = Math.max(1, Math.min(5, parseInt(framesTotalEl.value || '5', 10)));
  const everyN = Math.max(1, Math.min(10, parseInt(everyNEl.value || '2', 10)));
  regStatusEl.textContent = 'Capturing images...';
  capturedImages = [];
  let collected = 0, i = 0;
  while (collected < framesTotal) {
    if (i % everyN === 0) {
      capturedImages.push(snapshotDataUrl(videoReg || video));
      collected++;
    }
    i++;
    await new Promise(r => setTimeout(r, 80));
  }
  regStatusEl.textContent = `Captured ${capturedImages.length} frames.`;
  const nextBtn = document.getElementById('to-step-3');
  if (nextBtn) nextBtn.disabled = capturedImages.length === 0;
});

document.getElementById('to-step-3')?.addEventListener('click', () => {
  if (!capturedImages.length) { toast('Capture faces first'); return; }
  goToStep(3);
});

document.getElementById('btn-register')?.addEventListener('click', async () => {
  const aadhar = aadharEl.value.trim();
  if (!aadhar) return alert('Enter aadhar number first');
  if (!capturedImages.length) { toast('Please capture your face first'); return; }
  if (!/^\d{4,}$/.test(aadhar)) { toast('Enter numeric Aadhar'); return; }
  const rawName = nameEl.value.trim();
  if (!/^[A-Za-z ]{2,}$/.test(rawName)) { toast('Name must contain letters and spaces only'); return; }

  // Validate against registry before proceeding
  try {
    const validation = await (await fetch('/api/validate-registry', { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ aadhar, name: rawName }) 
    })).json();
    
    if (!validation.valid) {
      if (validation.error === 'not_in_registry') {
        toast('Aadhar not found in registry. Returning to face detection.');
        regStatusEl.textContent = 'Aadhar not in registry';
        returnToFaceDetection();
        return;
      } else if (validation.error === 'name_mismatch') {
        toast('Name does not match registry. Returning to face detection.');
        regStatusEl.textContent = 'Name mismatch with registry';
        returnToFaceDetection();
        return;
      }
      toast(validation.message || 'Validation failed');
      regStatusEl.textContent = 'Validation failed';
      returnToFaceDetection();
      return;
    }
  } catch (e) {
    toast('Registry validation failed. Returning to face detection.');
    regStatusEl.textContent = 'Registry validation failed';
    returnToFaceDetection();
    return;
  }

  regStatusEl.textContent = 'Uploading...';
  try {
    const name = rawName;
    const images = capturedImages;
    const res = await fetch('/api/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ aadhar, name, images }) });
    const js = await res.json();
    if (!res.ok) {
      if (js?.error === 'already_registered') {
        regStatusEl.textContent = 'This Aadhar is already registered. Please verify instead.';
        toast('Already registered. Verification required.');
        return;
      }
      if (js?.error === 'duplicate_face') {
        regStatusEl.textContent = `This face seems registered to ${js.registered_to}.`;
        alert('Duplicate: already registered to ' + js.registered_to);
        return;
      }
      throw new Error(js.error || 'Registration failed');
    }
    regStatusEl.textContent = `Registered ${js.added} frames for ${aadhar}`;
    toast('Registration successful. Review and continue.');
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
    // Step 7: pre-check if already voted
    const v = await (await fetch(`/api/vote/${encodeURIComponent(voter)}`)).json();
    if (v?.exists) {
      voteStatusEl.textContent = 'Already voted earlier.';
      toast(`Already voted: ${v.vote} on ${v.date} ${v.time}`);
      return;
    }
    const res = await fetch('/api/vote', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ voter, party }) });
    const js = await res.json();
    if (!res.ok) throw new Error(js.error || 'Vote failed');
    voteStatusEl.textContent = 'Vote recorded!';
    toast('Vote recorded');
    resetWizardAfterDelay();
  } catch (e) {
    voteStatusEl.textContent = e.message;
    alert(e.message);
  }
});

// Wizard navigation
function goToStep(n){
  [1,2,3,4,5].forEach(i => {
    const step = document.querySelector(`.step[data-step="${i}"]`);
    const content = document.getElementById(`step-${i}`);
    if (!step || !content) return;
    if (i === n) { step.classList.add('active'); content.hidden = false; }
    else { step.classList.remove('active'); content.hidden = true; }
  });
  if (n === 5) {
    confirmIdEl.textContent = aadharEl.value.trim() || lastLabel || '-';
    confirmNameEl.textContent = nameEl.value.trim() || '-';
    confirmPartyEl.textContent = partyEl.value || '-';
  }
}

document.getElementById('btn-next-3')?.addEventListener('click', () => {
  goToStep(4);
});
document.getElementById('btn-next-4')?.addEventListener('click', () => {
  goToStep(5);
});
document.getElementById('btn-back-5')?.addEventListener('click', () => {
  goToStep(4);
});

// Theme & accessibility
btnTheme?.addEventListener('click', () => {
  document.documentElement.classList.toggle('theme-light');
});
btnFontInc?.addEventListener('click', () => adjustFont(1));
btnFontDec?.addEventListener('click', () => adjustFont(-1));

function adjustFont(delta){
  const style = getComputedStyle(document.documentElement);
  const cur = parseFloat(style.fontSize || '16');
  const next = Math.max(12, Math.min(22, cur + delta));
  document.documentElement.style.fontSize = next + 'px';
}

// Kiosk mode
btnKiosk?.addEventListener('click', async () => {
  if (document.fullscreenElement) {
    await document.exitFullscreen();
  } else {
    await document.documentElement.requestFullscreen();
  }
  startInactivityTimer();
});

['click','keydown','mousemove','touchstart'].forEach(ev => {
  window.addEventListener(ev, () => startInactivityTimer());
});

function startInactivityTimer(){
  if (inactivityTimer) clearTimeout(inactivityTimer);
  inactivityTimer = setTimeout(() => {
    resetWizard();
    toast('Reset due to inactivity');
  }, 90_000); // 90s
}

function resetWizard(){
  lastLabel = null;
  labelEl.textContent = '-';
  aadharEl.value = '';
  nameEl.value = '';
  voteStatusEl.textContent = '';
  capturedImages = [];
  goToStep(1);
}

function returnToFaceDetection() {
  // Clear captured images and return to step 2 (face detection)
  capturedImages = [];
  aadharEl.value = '';
  nameEl.value = '';
  goToStep(2);
  toast('Please capture your face again and verify Aadhar details');
}

function resetWizardAfterDelay(){
  setTimeout(resetWizard, 4000);
}

// Toasts
function toast(msg){
  if (!toasts) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  toasts.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

init();



