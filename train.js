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


(async function() {
  const parser = new argparse.ArgumentParser();
  parser.addArgument('-g','--gpu', {
    action: 'storeTrue',
    help: 'Use tfjs-node-gpu for training (required CUDA and CuDNN)'
  });
  parser.addArgument('-f','--forget', {
    action: 'storeTrue',
    help: 'forget training for models that do not pass validation tests'
  });
  parser.addArgument('-l','--learningRate', {
    type: 'float',
    defaultValue: LEARNING_RATE,
    help: 'Learning rate used for training (default: 0.1)'
  });
  parser.addArgument('-v','--validationSplit', {
    type: 'float',
    defaultValue: VALIDATION_SPLIT,
    help: 'Amount of training data to withold for validation (default: 0.1)'
  });
  parser.addArgument('-e','--epochs', {
    type: 'int',
    defaultValue: TRAINING_EPOCHS,
    help: 'Number of epochs to train the model for'
  });
  parser.addArgument('-b','--batchSize', {
    type: 'int',
    defaultValue: TRAINING_BATCHSIZE,
    help: 'Batch size to be used during model training'
  });

  const args = parser.parseArgs();

  TRAINING_BATCHSIZE= args.batchSize;
  TRAINING_EPOCHS= args.epochs;
  VALIDATION_SPLIT = args.validationSplit;
  LEARNING_RATE = args.learningRate;
  RESTART_TRAINING_ON_VALIDATION_FAILURE = args.forget;

  if (args.gpu) {
    console.log('Training using GPU.');
    tf = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    tf = require('@tensorflow/tfjs-node');
  }

  const language = require('./language')(tf);

  await language.run(async ()=>{

      await language.load(true);

      let training_data = language.training_data;
      const languages = training_data.length;    
      
      
      training_data.forEach(t => {
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
        if (!t.model){
          newModel();
        } 
        // model was loaded, test to see if more training is needed
        else{
          const vd = language.validation_data.find(x=>x.language == t.language);
          if (vd){
            const result = language.detectTest(vd.text, vd.language);
          
            // model is predicting correctly
            if (result.pass){
              t.needsTraining = false;
              console.log(vd.language);
            }
            else {
              if (RESTART_TRAINING_ON_VALIDATION_FAILURE){
                newModel();
              }
            }
          }
        }

        t.model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', lr: LEARNING_RATE, metrics: ['acc']});
      });

      // only train the models that need it
      training_data = training_data.filter(x=>x.needsTraining);
      
      // find shortest length training text to use as the limit
      let shortest = training_data[0].text.length;
      for (let i=0; i < training_data.length; i++){
        if (training_data[i].text.length < shortest) shortest = training_data[i].text.length;
      }

      // for each training text, get 30 character sequences
      training_data.forEach((x,index)=>{
        const seq=[];
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
        x.seq = seq;
      });

      // build training data
      training_data.forEach((td,i) => {
        
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
              .slice(0, td.seq.length));  // take the same number of negative cases as positive cases

        // training outputs
        td.xy = array.create(td.seq.length, 1)
          .concat(array.create(td.seq.length,0));
      });

      const training_tasks=[];
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
          }}
        }));;
      });

      var t=setInterval( () => {
        const table = [];
        let i=0;
        let tmp=[];
        training_data.forEach(t=> {
          if (i < 2){
          tmp.push(t.language)

          let loss = t.logs.loss;
          if (loss <= 0.001)
            loss = color.green(loss.toFixed(4));
          else if (loss < 0.01)
            loss = color.yellow(loss.toFixed(4));
          else
            loss = color.red(loss.toFixed(4));
          tmp.push(loss);

          loss = t.logs.val_loss;
          if (loss <= 0.001)
            loss = color.green(loss.toFixed(4));
          else if (loss < 0.01)
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
        console.log(`Epoch ${training_data[0].epoch} of ${TRAINING_EPOCHS}`);
        console.log();
        console.log(columnify(table,{
          showHeaders: false,
          //minWidth: 15
          //columns: ['LANGUAGE', 'LOSS', 'VLOSS', 'LANGUAGE', 'LOSS', 'VLOSS'],
          config: {
            '2':{ minWidth: 15 },
            '5':{ minWidth: 15 },
            //'8':{ minWidth: 15 }
          }
        }));
      },1000);
      await Promise.all(training_tasks);
      clearInterval(t);

      training_data.forEach(async (t,i) => {
        await t.model.save(`file://./languagedetect/data/model/${t.language}`);
      });
  });

})();
