/* Run the same CSV analysis as the local server, entirely in the browser. */
const CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
let runtime;
async function getRuntime() {
  if (!runtime) runtime = (async () => {
    importScripts(CDN + 'pyodide.js');
    const py = await loadPyodide({indexURL: CDN});
    await py.loadPackage(['numpy', 'pandas', 'scikit-learn']);
    const response = await fetch(new URL('custom_dataset.py', self.location.href));
    if (!response.ok) throw new Error('無法載入分析程式');
    py.runPython(await response.text());
    return py;
  })().catch(error => { runtime = null; throw error; });
  return runtime;
}
let pending = Promise.resolve();
self.onmessage = ({data}) => {
  pending = pending.then(async () => {
    try {
      const py = await getRuntime();
      py.globals.set('_request_json', JSON.stringify(data.payload));
      const result = py.runPython(`import json\n_req = json.loads(_request_json)\njson.dumps(inspect_csv(_req['csv']) if _req['action'] == 'inspect' else train_csv(_req['csv'], _req['target'], _req['features']), ensure_ascii=False, allow_nan=False)`);
      self.postMessage({id:data.id, result:JSON.parse(result)});
    } catch (error) {
      self.postMessage({id:data.id, error: String(error.message || error).split('\n').filter(Boolean).at(-1)});
    }
  });
};
