const { train } = require('@tensorflow/tfjs-node');
const fs = require('fs');
let instance = {
  load,
  run,
  detect,
  detectTest,
  indexOfMax,
  arr,
  repeat,
  vectorize
};
let training_data;
let tf;

async function load(tryLoadSavedModels = true){
  training_data = JSON.parse(fs.readFileSync('./languagedetect/training-data.json','utf-8'));
  instance.training_data = training_data;


  if (tryLoadSavedModels){
    training_data.forEach(data => {
      data.model = await tf.loadLayersModel(`file://./languagedetect/model/${data.language}/${data.language}.json`);
    });
  }
  return instance;
}

function isPromise(obj){
  return obj !== undefined && typeof obj.then === 'function';
}
async function run( fn ){
  try{
    tf.engine().startScope();
    let res = fn();
    if (isPromise(res)){
      res = await Promise.resolve(res);
    }
    return res;
  }
  finally
  {
    tf.engine().endScope();
  }
}

function factory(tf_implementation)
{
  tf = tf_implementation;
  return instance;
}

function arr(size,index){
  let a= new Array(size);
  for (let i=0;i<size;i++){
    a[i]=0.0;
  }
  a[index]=1;
  return a;
}
function repeat(v,times){
  let a = [];
  for (let i=0; i<times; i++){
    a.push(v);
  }
  return a;
}
function vectorize(str){
  if (str.length < 30) str.padEnd(30,' ');
  let charCodes=[];
  for (let i = 0; i < 30; i++){
    charCodes.push(str.charCodeAt(i) / 65535);
  }
  return charCodes;
}
function indexOfMax(arr) {
  if (arr.length === 0) {
      return -1;
  }
  var max = arr[0];
  var maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }
  return maxIndex;
}

function detect(inputString){

  const results = training_data.map(data => {
    const result = data.model.predict(tf.tensor(vectorize(inputString)).reshape([1,30]));
    const result_prediction = result.arraySync()[0];
    return { language: data.language, confidence: result_prediction };
  });

  results.sort((a,b) => b.confidence - a.confidence);
  let max = results[0];
  max.input = inputString;
  
  return max;
}

function detectTest(
  /* string */        inputString, 
  /* function */  expectedLanguage
  ){
  const result = detect(inputString);
  const nc = "\x1b[0m"  
  const r = "\x1b[31m"
  const g = "\x1b[32m"
  
  if (expectedLanguage && typeof expectedLanguage === 'function')
  expectedLanguage = (()=> expectedLanguage({
    english: 'English',
    german: 'German',
    spanish: 'Spanish'
  }))();
  
  if (result.language == expectedLanguage){
    result.test = g + 'PASS' + nc;
  }
  else {
    result.test = r + 'FAIL' + nc;
  }

  return result;
}



module.exports = factory;