const LABELS = ["toxic", "offensive", "abusive"];

const el = (sel) => document.querySelector(sel);

function setThresh(val) {
  el('#thresh-val').textContent = Number(val).toFixed(2);
}

function chipClass(label) {
  switch (label) {
    case 'toxic': return 'chip bad';
    case 'offensive': return 'chip warn';
    case 'abusive': return 'chip bad';
    default: return 'chip';
  }
}

function renderChips(predicted) {
  const container = el('#labels');
  container.innerHTML = '';
  if (!predicted || predicted.length === 0) {
    const none = document.createElement('div');
    none.className = 'chip';
    none.textContent = 'No labels ≥ threshold';
    container.appendChild(none);
    return;
  }
  predicted.forEach(label => {
    const div = document.createElement('div');
    div.className = chipClass(label);
    div.textContent = label;
    container.appendChild(div);
  });
}

function renderBars(detailed) {
  const probs = el('#probs');
  probs.innerHTML = '';
  LABELS.forEach(label => {
    const p = detailed?.[label] ?? 0;
    const row = document.createElement('div');
    row.className = 'bar-row';
    const name = document.createElement('div');
    name.className = 'bar-label';
    name.textContent = label;
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    const val = document.createElement('div');
    val.className = 'bar-val';
    val.textContent = (p * 100).toFixed(1) + '%';
    // Append then animate for smoothness
    bar.appendChild(fill);
    row.appendChild(name);
    row.appendChild(bar);
    row.appendChild(val);
    probs.appendChild(row);
    requestAnimationFrame(() => { fill.style.width = Math.max(0, Math.min(100, p * 100)) + '%'; });
  });
}

async function doPredict() {
  const text = el('#text-input').value.trim();
  if (!text) {
    alert('Please enter text.');
    return;
  }
  const threshold = Number(el('#threshold').value || 0.5);

  el('#status').hidden = false;
  el('#output').hidden = true;

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, threshold })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.error || resp.statusText);
    renderChips(data.labels);
    renderBars(data.detailed);
    // Show threshold used (from server response to reflect clamping)
    el('#out-threshold').textContent = Number(data?.threshold ?? threshold).toFixed(2);
    el('#output').hidden = false;
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    el('#status').hidden = true;
  }
}

// Wiring
document.addEventListener('DOMContentLoaded', () => {
  setThresh(el('#threshold').value);
  el('#threshold').addEventListener('input', (e) => setThresh(e.target.value));
  el('#predict-btn').addEventListener('click', doPredict);
});
