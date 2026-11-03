/* Static Pages adapter: evaluate the exported fitted OLS locally in the browser. */
(function(){
  const original=window.fetch.bind(window);
  let worker, nextId=0; const requests=new Map();
  function customAnalysis(action,payload){
    if(!worker){
      worker=new Worker(new URL('custom_worker.js',document.baseURI));
      worker.onmessage=({data})=>{const pending=requests.get(data.id);if(!pending)return;requests.delete(data.id);clearTimeout(pending.timer);pending.resolve(data.error?Response.json({error:data.error},{status:400}):Response.json(data.result));};
      worker.onerror=()=>{for(const p of requests.values()){clearTimeout(p.timer);p.resolve(Response.json({error:'分析環境無法載入，請檢查網路後重新整理頁面。'},{status:503}));}requests.clear();worker.terminate();worker=null;};
    }
    return new Promise(resolve=>{const id=++nextId;const timer=setTimeout(()=>{requests.delete(id);resolve(Response.json({error:'分析環境載入逾時，請檢查網路並重新整理。'},{status:503}));},180000);requests.set(id,{resolve,timer});worker.postMessage({id,payload:{...payload,action}});});
  }
  window.fetch=async function(input,options={}){
    const path=typeof input==='string'?input:'';
    const custom=path.match(/^\/api\/custom\/(inspect|train)$/);
    if(custom){try{return await customAnalysis(custom[1],JSON.parse(options.body))}catch(err){return Response.json({error:err.message},{status:400})}}
    if(path==='/api/example') return Response.json(window.HW2_MODELS.life.example);
    const data=path.match(/^\/api\/datasets\/(paris|student)$/);
    if(data) return Response.json(window.HW2_MODELS[data[1]].info);
    const predict=path.match(/^\/api\/predict(?:\/(paris|student))?$/);
    if(predict){
      try{
        const model=window.HW2_MODELS[predict[1]||'life'];const record=JSON.parse(options.body);
        for(const f of [...model.numeric,...model.categorical])if(!(f.name in record))throw new Error('缺少欄位: '+f.name);
        return Response.json({prediction:window.predictStatic(model,record)});
      }catch(err){return Response.json({error:err.message},{status:400})}
    }
    return original(input,options);
  };
})();
