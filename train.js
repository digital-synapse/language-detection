let LOOP = false;
let RESTART_TRAINING_ON_VALIDATION_FAILURE = false;
let TRAINING_EPOCHS = 5000;
let LEARNING_RATE = 0.1;
let VALIDATION_SPLIT = 0.2;
let TRAINING_BATCHSIZE = undefined;//2500;

console.clear();
const argparse = require('argparse');
const array = require('./array-utils');
const columnify = require('columnify');
const color = require('colors-cli/safe');
let tf;
let language;

(async function() {
  const parser = new argparse.ArgumentParser();
  parser.addArgument(/*'-g',*/'--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)'
  });
  parser.addArgument(/*'-f',*/'--forget', {
    action: 'storeTrue',
    help: 'forget training for models that do not pass validation tests (if using with --loop, only forgets training on the first pass)'
  });
  parser.addArgument(/*'-l',*/'--learningRate', {
    type: 'float',
    defaultValue: LEARNING_RATE,
    help: `Learning rate used for training (default: ${LEARNING_RATE})`
  });
  parser.addArgument(/*'-v',*/'--validationSplit', {
    type: 'float',
    defaultValue: VALIDATION_SPLIT,
    help: `Amount of training data to withold for validation (default: ${VALIDATION_SPLIT})`
  });
  parser.addArgument(/*'-e',*/'--epochs', {
    type: 'int',
    defaultValue: TRAINING_EPOCHS,
    help: `Number of epochs to train the model for (default: ${TRAINING_EPOCHS})`
  });
  parser.addArgument(/*'-b',*/'--batchSize', {
    type: 'int',
    defaultValue: TRAINING_BATCHSIZE,
    help: 'Batch size to be used during model training (default: equal to the number of training samples)'
  });
  parser.addArgument('--loop', {
    action: 'storeTrue',
    help: 'Causes the training to loop until all models pass validation testing'
  });
  const args = parser.parseArgs();

  TRAINING_BATCHSIZE= args.batchSize;
  TRAINING_EPOCHS= args.epochs;
  VALIDATION_SPLIT = args.validationSplit;
  LEARNING_RATE = args.learningRate;
  RESTART_TRAINING_ON_VALIDATION_FAILURE = args.forget;
  LOOP = args.loop;

  if (args.gpu) {
    console.log('Training using GPU.');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    tf = require('@tensorflow/tfjs-node');
  }

  language = require('./language')(tf);

  await language.run(async ()=>{

    let loop =1;
    if (LOOP){
      while (await train(loop++));
    }
    else await train(1);

  });

})();

async function train(loop=1){
  
  await language.load(true);

  let training_data = language.training_data;
  const languages = training_data.length;    
  
  const init_model = t => {
    t.needsTraining=true;
    
    const newModel = () =>{
      const model = tf.sequential({ layers: [
        tf.layers.dense({ inputShape: [30], units: languages, activation: 'sigmoid' }),
        tf.layers.dense({ inputShape: [languages], units: languages, activation: 'sigmoid' }),
        tf.layers.dense({ inputShape: [languages], units: languages, activation: 'sigmoid' }),                             
        tf.layers.dense({ inputShape: [languages], units: 1, activation: 'sigmoid' })    
      ]});      
      t.model = model;
    };
    // no model (language.load() did not find a saved model to load, so create a new one)
    if (!t.model || t.forget){
      newModel();
    } 
    // model was loaded, test to see if more training is needed
    else{
      const vd = language.validation_data.find(x=>x.language == t.language);
      if (vd){
        const result = language.detectTest(vd.text, vd.language);
      
        // model is predicting correctly
        if (result.pass){
          if (t.forceAdditionalTraining){
            t.needsTraining = false;
            console.log(color.green(`${vd.language} : ${(result.confidence * 100).toFixed(1)}%`));
          }
        }
        else {
          if (RESTART_TRAINING_ON_VALIDATION_FAILURE && loop==1){
            console.log(color.yellow(`${vd.language} : Forgot training because forget option was specified and model failed validation test.`))
            newModel();
          }
          // if the prediction was wrong and highly confident,
          // save the negative result for training
          if (result.confidence > 0.9){
            t.falsePositive = result.predicted;
          }
        }
      }
    }

    t.model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', lr: LEARNING_RATE, metrics: ['acc']});
    t.compiled = true;
  };

  training_data.forEach(init_model);

  // only train the models that need it
  training_data = training_data.filter(x=>x.needsTraining);
  if (training_data.length == 0) return false;

  // find shortest length training text to use as the limit
  let shortest = training_data[0].text.length;
  for (let i=0; i < training_data.length; i++){
    if (training_data[i].text.length < shortest) shortest = training_data[i].text.length;
  }

  // for each training text, get 30 character sequences
  training_data.forEach((x,index)=>{
    
    // for languages that use spaces as word delimiters
    let seq=[];
    const tokens = x.text.split(' ');
    while (tokens.length > 0){
      let charCodes=[];
      let buffer='';
      for (let j=0; j<tokens.length; j++){
        buffer = buffer + tokens[j] + ' ';
        if (buffer.length >= 30) break;
      }
      for (let i = 0; i < 30; i++){
        charCodes.push(buffer.charCodeAt(i) / 65535);
      }
      tokens.pop();
      seq.push(charCodes)
    }

    // is this a language that uses spaces as sentence delimiters?
    if (seq.length < 50){
      seq=[];
      for (let j = 0; j < shortest - 30; j++){
        let charCodes=[];
        for (let i = 0; i < 30; i++){
          charCodes.push(x.text.charCodeAt(i+j) / 65535);
        }
        if (charCodes.filter(f=> Number.isNaN(f)).length > 0){
          throw new Error('NaN in training data');
        }
        seq.push(charCodes)
      }
    }
    x.seq = seq;
  });

  shortest = training_data[0].seq.length;
  for (let i=1; i < training_data.length; i++){
    if (training_data[i].seq.length < shortest) shortest = training_data[i].seq.length;
  }
  training_data.forEach((td,i) => {
    while (td.seq.length > shortest) td.seq.pop();
  });

  // build training data
  training_data.forEach((td,i) => {
    
    if (td.falsePositive){
      td.xx = td.seq.concat(training_data.filter(v=> v.language == td.falsePositive).seq);
    }
    
    if (td.xx && td.xx.length > shortest*2)
      td.xx = td.xx.slice(0,shortest*2);

    if ((td.xx && td.xx.length < shortest *2) || !td.xx){
      // training inputs
      td.xx = td.seq.concat(
        // for each sequence of positive test cases        
        array.shuffle(
        array.concatAll(
          training_data
            .filter((fv,fi)=> fi != i)  // select all datasets besides the current one
            .map(x=>x.seq)))            // select all their sequences
            //.concatAll()              // into a single array
            //.shuffle()                // shuffle the results to get a good random distribution
            .slice(0, (shortest *2)-td.seq.length));  // take the same number of negative cases as positive cases
    }

    // training outputs
    td.xy = array.create(td.seq.length, 1)
      .concat(array.create(td.seq.length,0));
  });

  const training_tasks=[];
  let startUpdates=false;
  training_data.forEach((t,i) => {
    training_tasks.push( t.model.fit(
      tf.tensor(t.xx), 
      tf.tensor(t.xy), {
      epochs: TRAINING_EPOCHS,
      batchSize: TRAINING_BATCHSIZE || (t.xx.length),
      shuffle: true,
      verbose: false,
      validationSplit: VALIDATION_SPLIT,
      callbacks: { onEpochEnd: (epoch, logs) =>{
        t.epoch = epoch;
        t.logs = logs;
        startUpdates=true;
      }}
    }));;
  });

  var t=setInterval( () => {
    if (!startUpdates) return;
    const table = [];
    let i=0;
    let tmp=[];
    training_data.forEach(t=> {
      if (i < 2){
      tmp.push(t.language)

      let loss = t.logs.loss;
      if (loss <= 0.01)
        loss = color.green(loss.toFixed(4));
      else if (loss < 0.2)
        loss = color.yellow(loss.toFixed(4));
      else
        loss = color.red(loss.toFixed(4));
      tmp.push(loss);

      loss = t.logs.val_loss;
      if (loss <= 0.01)
        loss = color.green(loss.toFixed(4));
      else if (loss < 0.2)
        loss = color.yellow(loss.toFixed(4));
      else
        loss = color.red(loss.toFixed(4));
      tmp.push(loss);

      i++;
      }
      else {
        i=0;
        table.push(tmp);
        tmp= [];
      }
    });
    if (tmp.length > 0) table.push(tmp);
    console.clear();
    console.log(`Epoch ${training_data[0].epoch} of ${TRAINING_EPOCHS} x ${loop}  LR: ${LEARNING_RATE}  VAL_SPLIT: ${VALIDATION_SPLIT}`);
    console.log();
    console.log(columnify(table,{
      showHeaders: false,
      //minWidth: 15
      //columns: ['LANGUAGE', 'LOSS', 'VLOSS', 'LANGUAGE', 'LOSS', 'VLOSS'],
      config: {
        '1':{ minWidth: 7 },
        '2':{ minWidth: 15 },
        '4':{ minWidth: 7 },
        '5':{ minWidth: 15 },
        //'8':{ minWidth: 15 }
      }
    }));
  },1000);
  await Promise.all(training_tasks);
  clearInterval(t);

  const save_tasks=[];
  training_data.forEach(async (t,i) => {
    const task = t.model.save(`file://./languagedetect/data/model/${t.language}`);
    save_tasks.push(task);
  });
  await Promise.all(save_tasks);

  return true;
}