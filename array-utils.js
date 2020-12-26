function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function concatAll(arrayOfArrays){
  let arr = []
  arr = arr.concat.apply(arr, arrayOfArrays);
  return arr;
}

function create(size,value){
  let a= new Array(size);
  for (let i=0;i<size;i++){
    a[i]=value;
  }
  return a;
}

module.exports = {
  shuffle,
  concatAll,
  create,
}