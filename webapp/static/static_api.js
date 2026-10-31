/* Static Pages adapter: evaluate the exported fitted OLS locally in the browser. */
(function(){
  const original=window.fetch.bind(window);
  window.fetch=async function(input,options={}){
    const path=typeof input==='string'?input:'';
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
