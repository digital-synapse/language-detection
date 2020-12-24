const tf = require('@tensorflow/tfjs-node');
const language = require('./language')(tf);

language.run(async ()=>{

  await language.load();

  const r = []
  console.clear();
  r.push(language.detectTest('مهلا ما الذي يحاول هذا النص قوله؟', 'Arabic'));  //arabic
  r.push(language.detectTest('hey wat probeer hierdie teks sê?', 'Afrikaans'));  //Afrikaans
  r.push(language.detectTest('oye, ¿qué trata de decir este texto?', 'Spanish')); //spanish
  r.push(language.detectTest('hé qu\'est-ce que ce texte essaie de dire?', 'French')); //french
  r.push(language.detectTest('hey what is this text trying to say?', 'English')); //english
  r.push(language.detectTest('Ei o le a le mea lea o loʻo taumafai le tala lea e fai?', 'Samoan')); //samoan
  r.push(language.detectTest('hey cosa sta cercando di dire questo testo?', 'Itallian')); // itallian
  r.push(language.detectTest('hej, kaj skuša povedati to besedilo?', 'Slovenian')); // slovenian
  r.push(language.detectTest('hei mitä tämä teksti yrittää sanoa?', 'Finnish')); // finnish
  r.push(language.detectTest('hei apa teks iki nyoba ngomong?', 'Javanese')); // javanese
  r.push(language.detectTest('hei, apa yang cuba disampaikan oleh teks ini?', 'Malay')); // malay  
  language.detectTest('', x=>x.
  console.table(r);

});