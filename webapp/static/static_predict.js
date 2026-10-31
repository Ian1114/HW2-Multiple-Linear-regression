(function (root) {
  function predictStatic(model, record) {
    let prediction = model.intercept;
    for (const f of model.numeric) {
      let value=record[f.name];
      if (f.name==='Status') value=({'Developing':0,'Developed':1})[value];
      if (value===null || value==='') value=f.median;
      value=Number(value);
      if (!Number.isFinite(value)) throw new Error(f.name+' 必須為有限數字');
      const options=model.info?.categories?.[f.name];
      if(options && !options.includes(value)) throw new Error(f.name+' 選項無效');
      prediction += ((value-f.mean)/f.scale)*f.coef;
    }
    for (const f of model.categorical) {
      let value=record[f.name];
      if (!f.options.includes(value)) throw new Error(f.name+' 選項無效');
      prediction += (value===f.positive ? 1 : 0)*f.coef;
    }
    return prediction;
  }
  if (typeof module !== 'undefined') module.exports={predictStatic};
  else root.predictStatic=predictStatic;
})(typeof window === 'undefined' ? globalThis : window);
