import { Dataset, Image, NeuralNetwork, prepare } from 'mnist-wasm'
import { memory } from "mnist-wasm/mnist_wasm_bg"

let trainButton = document.querySelector('#train')
trainButton.disabled = true

const WIDTH = 28
const HEIGHT = 28
const TRAINING_SIZE = 8000
const TESTING_SIZE = 2000

let canvas = document.querySelector('#image');
canvas.width = WIDTH
canvas.height = HEIGHT
let context = canvas.getContext('2d')

let currentImageIndicator = document.querySelector('#currentImage');
let image = 0;
let mnist = null;
let training = null;
let testing = null;

let mnistWasm = Dataset.new()
let network = NeuralNetwork.new()

let nextButton = document.querySelector('#next')
nextButton.disabled = true;
nextButton.addEventListener('click', () => {
    image += 1;
    drawCurrentImage()
})
let previousButton = document.querySelector('#previous')
previousButton.disabled = true;
previousButton.addEventListener('click', () => {
    image -= 1;
    drawCurrentImage()
})

let drawCurrentImage = () => {
    image = Math.min(Math.max(0, image), TRAINING_SIZE - 1)
    if (mnist && context) {
        mnist.draw(training.images[image], context)
    }
    currentImageIndicator.innerHTML = `Image #${image}: (${training.labels[image]})`
}

// The mnist package doesn't fetch the data over the network, it's actually
// just in memory. This is great for simplifying this example, but causes
// problems with the template setup of auto recompiling the JavaScript each
// time a file is modified. Moving the import to happen dynamically after
// the user clicks a button prevents the browser doing too much work in the
// background while editing files with the webserver open.
let prepareButton = document.querySelector('#prepare')
prepareButton.addEventListener('click', async () => {
    mnist = await import('mnist')

    let dataset = mnist.set(TRAINING_SIZE, TESTING_SIZE)

    training = splitData(dataset.training)
    testing = splitData(dataset.test)

    prepare()

    for (let i = 0; i < training.images.length; i++) {
        let image = training.images[i]
        let label = training.labels[i]
        let imageWasm = Image.new()
        let pixels = new Uint8Array(memory.buffer, imageWasm.buffer(), WIDTH * HEIGHT)
        // copy each pixel into the buffer exposed over Wasm to give it to
        // the Rust code
        for (let j = 0; j < WIDTH * HEIGHT; j++) {
            pixels[j] = image[j];
        }
        imageWasm.set_length()
        mnistWasm.add(imageWasm, label)
    }

    drawCurrentImage()

    nextButton.disabled = false;
    previousButton.disabled = false;
    trainButton.disabled = false;
    prepareButton.disabled = true;
})

/**
 * Converts a dataset provided by the mnist package into two seperate
 * arrays, the first, an array of images, and the second an array of labels.
 */
let splitData = (dataset) => {
    let labels = []
    let images = []
    for (let entry of dataset) {
        images.push(entry.input)
        // dataset is encoded as 1-hot, ie an image of 5 is represented as
        // [0 0 0 0 0 1 0 0 0 0], convert this to a single digit as data
        // transfer over wasm is slow
        labels.push(entry.output.indexOf(1))
    }
    return {
        labels: labels,
        images: images,
    }
}

// FIXME: Going to need to do training in a web worker
// https://developer.mozilla.org/en-US/docs/Web/API/Worker/postMessage

trainButton.addEventListener('click', () => {
    let loss = network.train(mnistWasm)
    console.log(`Average loss ${loss}`);
})
