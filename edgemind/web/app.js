'use strict';
const $ = (id) => document.getElementById(id);
let currentResult = null;
let activeTab = 'plan';
let serverConfig = null;

function labelHuman(value) {return String(value || '—').replaceAll('_', ' ');}
function textEl(tag, text, className='') {
  const node = document.createElement(tag);
  node.textContent = String(text ?? '');
  if (className) node.className = className;
  return node;
}
async function checkServer() {
  try {
    const response = await fetch('/api/health', {cache:'no-store'});
    if (!response.ok) throw Error('Server unavailable');
    serverConfig = await response.json();
    $('serverStatus').textContent = 'Local server connected';
    if (!serverConfig.real_local_enabled) {
      const option = $('executionMode').querySelector('[value=local_qwen]');
      option.textContent = 'Real local OpenVINO (enable on server)';
    }
  } catch (err) { $('serverStatus').textContent = 'Server unavailable'; }
}
function syncMode() {
  const isReal = $('executionMode').value === 'local_qwen';
  $('modeBadge').textContent = isReal ? 'REAL LOCAL' : 'MOCK MODE';
  $('modeBadge').classList.toggle('real', isReal);
  $('formNote').textContent = isReal
    ? 'The selected local OpenVINO model runs only on this machine. Generated code will not be executed.'
    : 'Mock mode only simulates AI generation, routing estimates and cloud eligibility. No real model is called.';
  if (isReal) {
    $('privacyMode').value = 'local_only';
    if ($('taskType').value === 'classification') $('taskType').value = 'text';
  }
  $('privacyMode').querySelector('[value=cloud_allowed]').disabled = isReal;
  $('taskType').querySelector('[value=classification]').disabled = isReal;
}
function requestBody() {
  const documentText = $('documentText').value.trim();
  return {
    query: $('query').value.trim(),
    execution_mode: $('executionMode').value === 'mock' ? 'mock' : $('localModel').value,
    task_type: $('taskType').value,
    privacy_mode: $('privacyMode').value,
    network: $('network').value,
    complexity: $('complexity').value,
    strategy: $('strategy').value,
    local_device: $('localDevice').value,
    max_new_tokens: Number($('maxTokens').value),
    rag_mode: $('ragMode').value,
    rag_top_k: Number($('ragTopK').value),
    rerank: $('rerank').checked,
    documents: documentText ? [{source_id:'user/input', text:documentText}] : [],
  };
}
function errorText(error) {
  if (typeof error === 'string') return error;
  if (Array.isArray(error)) return error.map(item => item.msg || JSON.stringify(item)).join('; ');
  return 'The request failed. Check the local server console.';
}
function displayTab(tab) {
  activeTab = tab;
  document.querySelectorAll('[data-tab]').forEach(b=>{
    const selected = b.dataset.tab === tab;
    b.classList.toggle('active', selected);
    b.setAttribute('aria-selected', String(selected));
  });
  const panel = $('tabPanel');
  panel.replaceChildren();
  if (!currentResult) return;
  if (tab === 'plan') {
    const plan = currentResult.plan || [];
    if (!plan.length) panel.appendChild(textEl('p','No plan returned.'));
    plan.forEach((step, i)=>{
      const node=textEl('div','', 'step');
      node.appendChild(textEl('strong', `${i+1}. ${labelHuman(step.operation)}`));
      node.appendChild(textEl('div', step.description || ''));
      node.appendChild(textEl('small',`Step ID: ${step.step_id || '?'} · Depends on: ${(step.depends_on||[]).join(', ')||'none'}`));
      panel.appendChild(node);
    });
    if ((currentResult.executed_steps||[]).length) panel.appendChild(textEl('p',`Executed: ${currentResult.executed_steps.join(' → ')}`));
  } else if (tab === 'trace') {
    const trace = currentResult.trace || [];
    if (!trace.length) panel.appendChild(textEl('p','No execution trace available.'));
    trace.forEach(line=>panel.appendChild(textEl('div',line,'trace-item')));
  } else {
    const details = {
      verification: currentResult.verification,
      model: currentResult.model,
      performance: currentResult.performance,
      evidence: currentResult.evidence,
      rag_diagnostics: currentResult.rag_diagnostics,
      benchmark_diagnostics: currentResult.benchmark_diagnostics,
      recovery_action: currentResult.recovery_action,
    };
    panel.appendChild(textEl('pre',JSON.stringify(details,null,2),'details-json'));
  }
}
function showResult(value, elapsed) {
  currentResult = value;
  $('emptyState').classList.add('hidden');
  $('resultContent').classList.remove('hidden');
  $('resultStatus').textContent = value.status || 'UNKNOWN';
  const okay = ['COMPLETED','SIMULATED'].includes(value.status);
  $('resultStatus').className = `result-status ${okay?'success':'failure'}`;
  $('resultNotice').textContent = value.notice || '';
  $('resultNotice').classList.toggle('warning',!value.is_real_inference);
  $('resultText').textContent = value.result || (value.error ? `No output: ${value.error}` : 'No output was generated.');
  $('metricModel').textContent = value.model?.model_id || 'None selected';
  $('metricDevice').textContent = value.model?.device || '—';
  const measured = value.performance?.workflow_total_ms;
  $('metricTime').textContent = typeof measured === 'number' ? `${(measured/1000).toFixed(2)}s observed` : `${(elapsed/1000).toFixed(2)}s request`;
  $('duration').textContent = `Finished in ${(elapsed/1000).toFixed(2)}s`;
  displayTab('plan');
}
async function run(event) {
  event.preventDefault();
  const body = requestBody();
  if (!body.query) { $('errorMessage').textContent = 'Please enter a task.'; $('errorMessage').classList.remove('hidden'); return; }
  $('runButton').disabled = true;
  $('runLabel').textContent = 'Running…';
  $('errorMessage').classList.add('hidden');
  $('duration').textContent = 'Working on your task…';
  const start=performance.now();
  try {
    const response = await fetch('/api/tasks', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw Error(errorText(data.detail));
    showResult(data, performance.now() - start);
  } catch (error) {
    $('errorMessage').textContent = error.message || 'Unexpected request error';
    $('errorMessage').classList.remove('hidden');
    $('duration').textContent = 'Request failed';
  } finally {
    $('runButton').disabled = false;
    $('runLabel').textContent = 'Run task';
  }
}
window.addEventListener('DOMContentLoaded',()=>{
  checkServer();
  syncMode();
  $('executionMode').addEventListener('change',syncMode);
  $('taskForm').addEventListener('submit',run);
  $('demoDocs').addEventListener('click',()=>{
    $('documentText').value = 'A GPU accelerates parallel computation for supported language models. An NPU is specialized for efficient neural-network inference. Private documents should remain on the local machine.';
  });
  $('ragMode').addEventListener('change',()=>{if ($('ragMode').value !== 'hybrid') $('rerank').checked=false;});
  $('copyOutput').addEventListener('click',async()=>{
    try { await navigator.clipboard.writeText($('resultText').textContent); $('copyOutput').textContent='Copied'; }
    catch (e) { $('copyOutput').textContent='Copy unavailable'; }
  });
  document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>displayTab(button.dataset.tab)));
});
