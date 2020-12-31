const fs = require('fs');
let instance = {
  load,
  run,
  arr_init,
  detect,
  detectTest,
  indexOfMax,
  arr,
  repeat,
  vectorize,
  loadJSON
};
let training_data;
let tf;

function loadJSON(filePath){
  return JSON.parse(fs.readFileSync(filePath,'utf-8'));
}
async function load(tryLoadSavedModels = true){
  training_data = loadJSON('./languagedetect/data/training-data-1.json');
  const td2 = loadJSON('./languagedetect/data/training-data-2.json');
  const td3 = loadJSON('./languagedetect/data/training-data-3.json');
  const td4 = loadJSON('./languagedetect/data/training-data-4.json');
  const td5 = loadJSON('./languagedetect/data/training-data-5.json');
  training_data.forEach(d=> {
    const a= td2.find(x=>x.language == d.language);
    if (a && a.text){
      d.text = d.text + ' ' + a.text;
    }
    const b= td3.find(x=>x.language == d.language);
    if (b && b.text){
      d.text = d.text + ' ' + b.text;
    }
    const c= td4.find(x=>x.language == d.language);
    if (c && c.text){
      d.text = d.text + ' ' + c.text;
    }
    const aa= td5.find(x=>x.language == d.language);
    if (aa && aa.text){
      d.text = d.text + ' ' + aa.text;
    }
  });
  instance.training_data = training_data;
  instance.validation_data = loadJSON('./languagedetect/data/validation-data.json');

  if (tryLoadSavedModels){
    for (let i=0; i<training_data.length; i++){    
      try{
        training_data[i].model = await tf.loadLayersModel(`file://./languagedetect/data/model/${training_data[i].language}/model.json`);
      }
      catch {
        console.log(`warning: model data not found for ${training_data[i].language}`);        
      }
    };
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

function arr_init(size,value){
  let a= new Array(size);
  for (let i=0;i<size;i++){
    a[i]=value;
  }
  return a;
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
function normalizeLen(str, len = 30){  
  if (str.length < len) str = str.padEnd(len,' ');
  if (str.length > len) str = str.substring(0,len);
  return str;
}
function vectorize(str){
  if (str.length < 30) str = str.padEnd(30,' ');
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

function predict(data, inputString){
  if (data.model){
    const result = data.model.predict(tf.tensor(vectorize(inputString)).reshape([1,30]));
    let result_prediction = result.arraySync()[0][0];
    if (isNaN(result_prediction)){
      result_prediction= 0.0;
      return { predicted: data.language, confidence: result_prediction, error: 'NaN' };
    }
    return { predicted: data.language, confidence: result_prediction };
  }
  else {
    return  { predicted: data.language, confidence: 0, error: 'model was missing' };
  }
}

function detect(inputString){

  const results = training_data.map(data => predict(data,inputString));

  results.sort((a,b) => b.confidence - a.confidence);
  let max = results[0];
  max.input = normalizeLen(inputString);
  
  return max;
}

function detectTest(inputString, expectedLanguage)
{
  const result = detect(inputString);
  const detail = predict(training_data.find(x=>x.language == expectedLanguage), inputString);
  result.expected = expectedLanguage;
  result.detail = detail;
  if (result.predicted == expectedLanguage && result.confidence >= 0.5){
    result.pass =true;
  }
  else {
    result.pass =false;
  }

  return result;
}



module.exports = factory;