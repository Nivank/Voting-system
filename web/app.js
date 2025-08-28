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
const btnTheme = document.getElementById('btn-theme');
const btnFontInc = document.getElementById('font-inc');
const btnFontDec = document.getElementById('font-dec');
const btnVoice = document.getElementById('btn-voice');
const btnKiosk = document.getElementById('btn-kiosk');
const toasts = document.getElementById('toasts');
const btnNew = document.getElementById('btn-new');

let lastLabel = null;
let voiceOn = true;
let inactivityTimer = null;

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
  speak('Welcome. Step one: Identify yourself and press Identify.');
}

function snapshotDataUrl() {
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
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
        speak('We could not identify you. You can register as a new user.');
        goToStep(2);
        expandRegistration();
        return;
      }
      throw new Error(js.error || 'Prediction failed');
    }
    lastLabel = js.label;
    labelEl.textContent = js.label;
    aadharEl.value = js.label;
    goToStep(2);
    toast('Identified as ' + js.label);
    speak('Identification complete. Please verify your details.');
  } catch (e) {
    labelEl.textContent = '-';
    toast(e.message);
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
    goToStep(3);
    toast('Registration successful');
    speak('Registration successful. Please choose your party.');
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
    toast('Vote recorded');
    speak('Your vote has been recorded. Thank you.');
    resetWizardAfterDelay();
  } catch (e) {
    voteStatusEl.textContent = e.message;
    alert(e.message);
  }
});

// Wizard navigation
function goToStep(n){
  [1,2,3,4].forEach(i => {
    const step = document.querySelector(`.step[data-step="${i}"]`);
    const content = document.getElementById(`step-${i}`);
    if (!step || !content) return;
    if (i === n) { step.classList.add('active'); content.hidden = false; }
    else { step.classList.remove('active'); content.hidden = true; }
  });
  if (n === 4) {
    confirmIdEl.textContent = aadharEl.value.trim() || lastLabel || '-';
    confirmPartyEl.textContent = partyEl.value || '-';
  }
}

function expandRegistration(){
  const details = document.querySelector('#step-2 details');
  if (details) details.open = true;
}

document.getElementById('btn-next-2')?.addEventListener('click', () => {
  goToStep(3);
  speak('Please choose your party.');
});
btnNew?.addEventListener('click', () => {
  goToStep(2);
  expandRegistration();
  speak('Please enter your details and start registration.');
});
document.getElementById('btn-next-3')?.addEventListener('click', () => {
  goToStep(4);
  speak('Review and press submit to confirm your vote.');
});
document.getElementById('btn-back-4')?.addEventListener('click', () => {
  goToStep(3);
});

// Theme & accessibility
btnTheme?.addEventListener('click', () => {
  document.documentElement.classList.toggle('theme-light');
});
btnFontInc?.addEventListener('click', () => adjustFont(1));
btnFontDec?.addEventListener('click', () => adjustFont(-1));
btnVoice?.addEventListener('click', () => {
  voiceOn = !voiceOn;
  btnVoice.textContent = 'Voice: ' + (voiceOn ? 'On' : 'Off');
  if (voiceOn) speak('Voice guidance enabled');
});

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
  goToStep(1);
}
function resetWizardAfterDelay(){
  setTimeout(resetWizard, 4000);
}

// Toasts & voice
function toast(msg){
  if (!toasts) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  toasts.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
function speak(text){
  if (!voiceOn || !('speechSynthesis' in window)) return;
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'en-US';
  window.speechSynthesis.speak(utt);
}

init();



