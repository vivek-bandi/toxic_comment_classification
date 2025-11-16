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

function renderBars(detailed, confidence, uncertainty) {
  const probs = el('#probs');
  probs.innerHTML = '';
  LABELS.forEach(label => {
    const p = detailed?.[label] ?? 0;
    const conf = confidence?.[label] ?? 1.0;
    const unc = uncertainty?.[label] ?? 0.0;
    
    const row = document.createElement('div');
    row.className = 'bar-row';
    
    const name = document.createElement('div');
    name.className = 'bar-label';
    name.textContent = label;
    
    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    
    // Add confidence indicator as overlay
    const confIndicator = document.createElement('div');
    confIndicator.className = 'confidence-indicator';
    confIndicator.style.width = (conf * 100) + '%';
    
    const val = document.createElement('div');
    val.className = 'bar-val';
    val.innerHTML = `${(p * 100).toFixed(1)}% <span class="conf-badge" title="Confidence: ${(conf * 100).toFixed(0)}%">±${(unc * 100).toFixed(1)}%</span>`;
    
    // Append then animate for smoothness
    bar.appendChild(fill);
    bar.appendChild(confIndicator);
    row.appendChild(name);
    row.appendChild(bar);
    row.appendChild(val);
    probs.appendChild(row);
    requestAnimationFrame(() => { 
      fill.style.width = Math.max(0, Math.min(100, p * 100)) + '%';
      confIndicator.style.opacity = '0.3';
    });
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
    const ct = resp.headers.get('content-type') || '';
    let data;
    if (ct.includes('application/json')) {
      data = await resp.json();
      if (!resp.ok) throw new Error(data?.error || resp.statusText);
    } else {
      const textBody = await resp.text();
      throw new Error(`Non-JSON response (${resp.status}): ${textBody.slice(0, 200)}`);
    }
    renderChips(data.labels);
    renderBars(data.detailed, data.confidence, data.uncertainty);
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
