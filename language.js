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
    for (let i=0; i<training_data.length; i++){    
      training_data[i].model = await tf.loadLayersModel(`file://./languagedetect/model/${training_data[i].language}/model.json`);
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
    return { predicted: data.language, confidence: result_prediction };
  });

  results.sort((a,b) => b.confidence - a.confidence);
  let max = results[0];
  max.input = inputString;
  
  return max;
}

function detectTest(inputString, expectedLanguage)
{
  const result = detect(inputString);
  result.expected = expectedLanguage;
  if (result.predicted == expectedLanguage){
    result.pass =true;
  }
  else {
    result.pass =false;
  }

  return result;
}



module.exports = factory;