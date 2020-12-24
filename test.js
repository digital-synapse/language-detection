const columnify = require('columnify');
const color = require('colors-cli/safe');
const tf = require('@tensorflow/tfjs-node');
const language = require('./language')(tf);

console.clear();

language.run(async ()=>{
  await language.load();
  const r = []
  const data = language.loadJSON('./languagedetect/data/validation-data.json');

  data.forEach(d => {
    r.push(language.detectTest(d.text, d.language));    
  });

  r.forEach(o=> {
    if (o.pass) o.pass = color.green.bold('PASS');
    else o.pass = color.red.bold('FAIL');

    o.confidence = o.confidence.toFixed(4);
    if (o.confidence < 0.5) o.confidence = color.yellow(o.confidence);    
    
  });
  console.log(columnify(r, {
    columns: ['predicted','confidence','expected','pass'/*, 'input'*/],
    headingTransform: (text) => {
      const repeat = () => `${'-'.repeat(text.length)}`;
      return `${text.toUpperCase()}\n${repeat()}`
  }
  }));

});