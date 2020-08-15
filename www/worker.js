import { Dataset, Image, NeuralNetwork, prepare } from 'mnist-wasm'
import { memory } from 'mnist-wasm/mnist_wasm_bg'

const WIDTH = 28
const HEIGHT = 28
const TRAINING_SIZE = 8000
const TESTING_SIZE = 2000

let mnist = null
let training = null
let testing = null

let mnistWasm = Dataset.new()
let network = NeuralNetwork.new()

onmessage = async (event) => {
    let data = event.data
    if (data.prepareDataset) {
        mnist = await import('mnist')

        let dataset = mnist.set(TRAINING_SIZE, TESTING_SIZE)
        training = splitData(dataset.training)
        testing = splitData(dataset.test)

        for (let i = 0; i < training.images.length; i++) {
            let image = training.images[i]
            let label = training.labels[i]
            let imageWasm = Image.new()
            let pixels = new Uint8Array(memory.buffer, imageWasm.buffer(), WIDTH * HEIGHT)
            // copy each pixel into the buffer exposed over Wasm to give it to
            // the Rust code
            for (let j = 0; j < WIDTH * HEIGHT; j++) {
                pixels[j] = image[j]
            }
            imageWasm.set_length()
            mnistWasm.add(imageWasm, label)
        }
        postMessage({ datasetPrepared: true })
    }
    if (data.trainEpoch) {
        let loss = network.train(mnistWasm)
        postMessage({ trainedEpoch: true, loss: loss })
    }
    if (data.requestCurrentImage) {
        image = Math.min(Math.max(0, data.currentImage), TRAINING_SIZE - 1)
        postMessage({
            currentImage: true,
            imageData: training.images[image],
            label: training.labels[image],
            index: image
        })
    }
}

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
