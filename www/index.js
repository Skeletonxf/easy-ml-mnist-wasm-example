let trainButton = document.querySelector('#train')
trainButton.disabled = true

const WIDTH = 28
const HEIGHT = 28
const TRAINING_SIZE = 8000
const TESTING_SIZE = 2000

let canvas = document.querySelector('#image')
canvas.width = WIDTH
canvas.height = HEIGHT
let context = canvas.getContext('2d')

let currentImageIndicator = document.querySelector('#currentImage')
let image = 0

let nextButton = document.querySelector('#next')
nextButton.disabled = true
nextButton.addEventListener('click', () => {
    image += 1
    drawCurrentImage()
})
let previousButton = document.querySelector('#previous')
previousButton.disabled = true
previousButton.addEventListener('click', () => {
    image -= 1
    drawCurrentImage()
})

let worker = new Worker('worker.js', { type: 'module' })

let drawCurrentImage = () => {
    worker.postMessage({ requestCurrentImage: true, currentImage: image })
}

let prepareButton = document.querySelector('#prepare')
prepareButton.addEventListener('click', async () => {
    worker.postMessage({ prepareDataset: true })
})

trainButton.addEventListener('click', () => {
    worker.postMessage({ trainEpoch: true })
})

worker.onmessage = (event) => {
    let data = event.data
    if (data.datasetPrepared) {
        drawCurrentImage()

        nextButton.disabled = false
        previousButton.disabled = false
        trainButton.disabled = false
        prepareButton.disabled = true
    }
    if (data.currentImage) {
        let image = data.imageData
        let label = data.label
        let index = data.index
        // TODO: Draw image data to canvas again
        console.log(image)
        currentImageIndicator.innerHTML = `Image #${index}: (${label})`
    }
    if (data.trainedEpoch) {
        console.log(`Loss: ${data.loss}`)
    }
}
