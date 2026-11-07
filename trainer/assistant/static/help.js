// copy buttons on all command blocks
document.querySelectorAll('pre.cmd').forEach(pre => {
  const b = document.createElement('button');
  b.className = 'copy'; b.type = 'button'; b.textContent = 'copy';
  b.onclick = () => {
    const text = pre.innerText.replace(/copy$|copied ✓$/, '').trim();
    navigator.clipboard.writeText(text).then(() => { b.textContent = 'copied ✓'; setTimeout(() => b.textContent = 'copy', 1500); });
  };
  pre.appendChild(b);
});

async function postJSON(url, data) {
  const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data || {})});
  return r.json();
}
