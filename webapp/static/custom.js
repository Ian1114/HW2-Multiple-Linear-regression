const customFile = document.querySelector('#custom-file');
const customStatus = document.querySelector('#custom-status');
const customTarget = document.querySelector('#custom-target');
const customFeatures = document.querySelector('#custom-features');
const customTrain = document.querySelector('#custom-train');
const customRemove = document.querySelector('#custom-remove');
let customCsv = '', customColumns = [], customVersion = 0;

async function customRequest(path, data) {
  const response = await fetch('/api/custom/' + path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || '分析失敗，請檢查資料。');
  return body;
}
function featureList() {
  customFeatures.replaceChildren();
  for (const col of customColumns.filter(c => c.name !== customTarget.value)) {
    const label = document.createElement('label');
    const box = document.createElement('input'); box.type = 'checkbox'; box.value = col.name; box.checked = col.missing < customRowCount && (col.numeric || col.unique <= 100);
    const text = document.createElement('span'); text.textContent = `${col.name} · ${col.numeric ? '數值' : '類別'}${col.missing ? ` · 缺值 ${col.missing}` : ''}${!col.numeric && col.unique > 100 ? ' · 類別過多，已取消選取' : ''}`;
    label.append(box, text); customFeatures.append(label);
  }
  document.querySelector('#custom-results').hidden = true;
}
customTarget.addEventListener('change', featureList);
let customRowCount = 0;
function clearCustom() {
  ++customVersion; customFile.value = ''; customCsv = ''; customColumns = []; customRowCount = 0;
  customTarget.replaceChildren(); customFeatures.replaceChildren(); customTrain.disabled = true;
  customTarget.disabled = false; document.querySelector('#custom-select-all').disabled = false; document.querySelector('#custom-clear-all').disabled = false;
  customRemove.hidden = true; document.querySelector('#custom-config').hidden = true;
  document.querySelector('#custom-results').hidden = true; customStatus.textContent = '尚未選擇檔案。';
}
customRemove.addEventListener('click', clearCustom);
customFile.addEventListener('change', async () => {
  const version = ++customVersion, file = customFile.files[0];
  customTarget.disabled = false; document.querySelector('#custom-select-all').disabled = false; document.querySelector('#custom-clear-all').disabled = false;
  customCsv = ''; customColumns = []; customTrain.disabled = true;
  document.querySelector('#custom-config').hidden = true; document.querySelector('#custom-results').hidden = true;
  if (!file) { customStatus.textContent = '尚未選擇檔案。'; return; }
  customRemove.hidden = false;
  if (file.size > 8 * 1024 * 1024) { customStatus.textContent = '檔案不可超過 8 MB。'; return; }
  customStatus.textContent = '正在檢查資料…';
  try {
    const csv = await file.text(), info = await customRequest('inspect', {csv});
    if (version !== customVersion) return;
    customCsv = csv; customColumns = info.columns; customRowCount = info.rows; customTarget.replaceChildren();
    for (const col of info.columns.filter(c => c.numeric)) { const option = document.createElement('option'); option.value = col.name; option.textContent = col.name; customTarget.append(option); }
    if (!customTarget.options.length) throw new Error('找不到數值目標欄位，請確認資料內容。');
    customTarget.selectedIndex = customTarget.options.length - 1;
    featureList(); customStatus.textContent = `${file.name} · ${info.rows.toLocaleString()} 筆資料 · ${info.columns.length} 個欄位`;
    document.querySelector('#custom-config').hidden = false; customTrain.disabled = false;
  } catch (error) { if (version === customVersion) customStatus.textContent = error.message || '無法讀取檔案，請確認為 UTF-8 CSV。'; }
});
document.querySelector('#custom-select-all').addEventListener('click', () => { customFeatures.querySelectorAll('input').forEach(box => box.checked = true); document.querySelector('#custom-results').hidden = true; });
document.querySelector('#custom-clear-all').addEventListener('click', () => { customFeatures.querySelectorAll('input').forEach(box => box.checked = false); document.querySelector('#custom-results').hidden = true; });
customFeatures.addEventListener('change', () => { document.querySelector('#custom-results').hidden = true; });
customTrain.addEventListener('click', async () => {
  const version = customVersion, target = customTarget.value, features = [...customFeatures.querySelectorAll('input:checked')].map(box => box.value);
  customTrain.disabled = true; customTarget.disabled = true; customFeatures.querySelectorAll('input').forEach(box => box.disabled = true); document.querySelector('#custom-select-all').disabled = true; document.querySelector('#custom-clear-all').disabled = true;
  customStatus.textContent = '正在訓練模型及計算交叉驗證，請稍候…'; document.querySelector('#custom-results').hidden = true;
  try {
    const result = await customRequest('train', {csv:customCsv, target, features});
    if (version !== customVersion) return;
    const fmt = n => Number(n).toLocaleString('zh-TW', {maximumFractionDigits:4});
    for (const name of ['mae','mse','rmse','r2']) {
      document.querySelector('#custom-test-' + name).textContent = fmt(result.test[name]);
      document.querySelector('#custom-train-' + name).textContent = fmt(result.train[name]);
    }
    document.querySelector('#custom-summary').textContent = `目標：${result.target}｜特徵：${result.features.length} 個（數值 ${result.numeric.length}、類別 ${result.categorical.length}）｜有效資料 ${result.rows.toLocaleString()} 筆，目標缺失排除 ${result.dropped} 筆。訓練 ${result.train_rows} 筆、測試 ${result.test_rows} 筆。`;
    document.querySelector('#custom-cv').textContent = `訓練集五折交叉驗證平均 RMSE：${fmt(result.cv_rmse)}；測試集平均值基準 RMSE：${fmt(result.baseline.rmse)}。`;
    document.querySelector('#custom-warning').textContent = result.features.length < 14 || result.features.length > 20 ? '目前選擇的特徵數不在作業建議的 14–20 個範圍內；請依資料實際情況調整，並檢查識別碼與資料洩漏。' : '請檢查識別碼、預測時不可取得的欄位與資料洩漏；相關不代表因果。';
    document.querySelector('#custom-results').hidden = false; customStatus.textContent = '分析完成。';
    document.querySelector('#custom-results').scrollIntoView({behavior:'smooth'});
  } catch (error) { if (version === customVersion) customStatus.textContent = error.message || '分析失敗。'; }
  finally { if (version === customVersion) { customTrain.disabled = false; customTarget.disabled = false; customFeatures.querySelectorAll('input').forEach(box => box.disabled = false); document.querySelector('#custom-select-all').disabled = false; document.querySelector('#custom-clear-all').disabled = false; } }
});
