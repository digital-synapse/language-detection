# Language Detection

### To train the language detection models run

    node train.js

if you dont have an nvidia GPU installed with CUDA, you may see errors when trying to build the models. If this happens, change the import in train.js from this:

    const tf = require('@tensorflow/tfjs-node-gpu');

to this:

    const tf = require('@tensorflow/tfjs-node');

### After training the models you can run some test predictions

    node test.js