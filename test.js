const tf = require('@tensorflow/tfjs-node');
const language = require('./language')(tf);

language.run(async ()=>{

  await language.load();

  const r = []
  console.clear();
  const data = language.loadJSON('./languagedetect/data/validation-data.json');

  data.forEach(d => {
    r.push(language.detectTest(d.text, d.language));    
  });

  console.table(r);

});