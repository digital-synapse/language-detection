console.clear();
const columnify = require('columnify');
const tf = require('@tensorflow/tfjs-node-gpu');
const language = require('./language')(tf);
let training_epochs = 5000;

language.run(async ()=>{

    await language.load(true);

    let training_data = language.training_data;
    const languages = training_data.length;    
    
    
    training_data.forEach(t => {
      t.needsTraining=true;
      
      // no model (language.load() did not find a saved model to load, so create a new one)
      if (!t.model){
        const model = tf.sequential({ layers: [
          tf.layers.dense({ inputShape: [30], units: languages, activation: 'sigmoid' }),
          tf.layers.dense({ inputShape: [languages], units: languages, activation: 'sigmoid' }),
          tf.layers.dense({ inputShape: [languages], units: languages, activation: 'sigmoid' }),
          tf.layers.dense({ inputShape: [languages], units: 1, activation: 'sigmoid' })    
        ]});      
        t.model = model;
      } 
      // model was loaded, test to see if more training is needed
      else{
        training_epochs = 500;
        const vd = language.validation_data.find(x=>x.language == t.language);
        if (vd){
          const result = language.detectTest(vd.text, vd.language);
        
          // model is predicting correctly
          if (result.pass){
            t.needsTraining = false;
          }
        }
      }

      t.model.compile({ optimizer: 'sgd', loss: 'binaryCrossentropy', lr: 0.001 });
    });

    // only train the models that need it
    training_data = training_data.filter(x=>x.needsTraining);

    let shortest = training_data[0].text.length;
    for (let i=0; i < training_data.length; i++){
      if (training_data[i].text.length < shortest) shortest = training_data[i].text.length;
    }
    training_data.forEach((x,index)=>{
      const seq=[];
      if (x.text.length < shortest){
        throw new Error('Training text length was smaller than expected');
      }
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
      x.xx = seq;
    });
    const xx = [].concat.apply([], training_data.map(x=>x.xx));

    const xy = [];
    for (let z=0; z<training_data.length; z++){
      const tmp = [];
      for (let j=0; j<training_data.length; j++){
        for (let i=0; i<training_data[j].xx.length; i++){
          if (j==z) tmp.push(1);
          else tmp.push(0);
        }
      }
      xy.push(tmp);
    }

    const training_tasks=[];
    training_data.forEach((t,i) => {
      training_tasks.push( t.model.fit(tf.tensor(xx), tf.tensor(xy[i]), {
        epochs: training_epochs,
        batchSize: xx.length,
        shuffle: true,
        verbose: false,
        validationSplit: 0.1,
        callbacks: { onYield: (epoch, batch, logs) =>{
          t.epoch = epoch;
          t.batch = batch;
          t.logs = logs;
        }}
      }));;
    });

    var t=setInterval( () => {
      const table = [];
      let i=0;
      let tmp=[];
      training_data.forEach(t=> {
        if (i < 3){
        tmp.push(t.language)
        tmp.push(t.logs.loss.toFixed(5));
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
      console.log(`Epoch ${training_data[0].epoch} of ${training_epochs}`)
      console.log(columnify(table));
    },1000);
    await Promise.all(training_tasks);
    clearInterval(t);

    training_data.forEach(async (t,i) => {
      await t.model.save(`file://./languagedetect/data/model/${t.language}`);
    });
});