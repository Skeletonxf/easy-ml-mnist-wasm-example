importScripts('pkg/mnist_wasm.js')
// hack to make mnist.js load correctly in a web worker environment
var window = self
importScripts('data/mnist.js') // FIXME: We should be able to programmatically copy this file out of node_modules instead of vendoring it

const WIDTH = 28
const HEIGHT = 28
const TRAINING_SIZE = 8000
const TESTING_SIZE = 2000

let training = null
let testing = null

let memory = null

function logProgress(percent) {
    postMessage({
        progress: true,
        percent: percent
    })
}

function logBatchLoss(percent) {
    postMessage({
        batchLoss: true,
        percent: percent
    })
}

wasm_bindgen('pkg/mnist_wasm_bg.wasm').then(mnistWasmModule => {
    memory = mnistWasmModule.memory
    const { Dataset, Image, NeuralNetwork } = wasm_bindgen

    let trainingDataset = Dataset.new_training()
    let testingDataset = Dataset.new_testing()
    let network = NeuralNetwork.new()

    let intoImage = (image) => {
        let imageWasm = Image.new()
        let pixels = new Float64Array(memory.buffer, imageWasm.buffer(), WIDTH * HEIGHT)
        // copy each pixel into the buffer exposed over Wasm to give it to
        // the Rust code
        for (let j = 0; j < WIDTH * HEIGHT; j++) {
            pixels[j] = image[j]
        }
        imageWasm.set_length()
        return imageWasm
    }

    onmessage = async (event) => {
        let data = event.data
        if (data.prepareDataset) {
            let dataset = mnist.set(TRAINING_SIZE, TESTING_SIZE)
            training = splitData(dataset.training)
            testing = splitData(dataset.test)

            for (let i = 0; i < training.images.length; i++) {
                let image = training.images[i]
                let label = training.labels[i]
                trainingDataset.add(intoImage(image), label)
            }

            for (let i = 0; i < testing.images.length; i++) {
                let image = testing.images[i]
                let label = testing.labels[i]
                testingDataset.add(intoImage(image), label)
            }

            postMessage({ datasetPrepared: true })
            postAccuracy()
        }
        if (data.trainEpoch) {
            let loss = network.train(trainingDataset)
            postMessage({ trainedEpoch: true, loss: loss })
            postAccuracy()
        }
        if (data.requestCurrentImage) {
            image = Math.min(Math.max(0, data.currentImage), TRAINING_SIZE - 1)
            let classification = network.classify(intoImage(training.images[image]))
            postMessage({
                currentImage: true,
                imageData: training.images[image],
                label: training.labels[image],
                index: image,
                classification: classification
            })
        }
    }

    postMessage({
        loadedWorker: true
    })

    let postAccuracy = () => {
        let trainingAccuracy = network.accuracy(trainingDataset)
        let testingAccuracy = network.accuracy(testingDataset)
        postMessage({
            accuracy: true,
            trainingAccuracy: trainingAccuracy,
            testingAccuracy: testingAccuracy
        })
    }
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
