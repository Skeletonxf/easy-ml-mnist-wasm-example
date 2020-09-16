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

let drawNegative = false

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
prepareButton.disabled = true
prepareButton.textContent = 'Loading'
prepareButton.addEventListener('click', () => {
    worker.postMessage({ prepareDataset: true })
})

trainButton.addEventListener('click', () => {
    worker.postMessage({ trainEpoch: true })
    trainButton.textContent = 'Training (This may take some time)'
    trainButton.disabled = true
    nextButton.disabled = true
    previousButton.disabled = true
    drawMode.disabled = true
})

let drawMode = document.querySelector('#viewMode')
drawMode.addEventListener('change', () => {
    drawNegative = !drawNegative
    drawCurrentImage()
})

let progressLabel = document.querySelector('label[for=trainingProgress]')
let progressBar = document.querySelector('#trainingProgress')

let chartList = document.querySelector('figure.chart ul')
let pointsPlotted = 0

let trainingAccuracy = document.querySelector('#trainingAccuracy')
let testingAccuracy = document.querySelector('#testingAccuracy')

let getColor = (color) => {
    if (drawNegative) {
        return `rgb(${color * 255}, ${color * 255}, ${color * 255})`
    } else {
        return `rgb(${255 - (color * 255)}, ${255 - (color * 255)}, ${255 - (color * 255)})`
    }
}

worker.onmessage = (event) => {
    let data = event.data
    if (data.loadedWorker) {
        prepareButton.disabled = false
        prepareButton.textContent = 'Prepare Data'
    }
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
        // Draw image data to canvas
        let color = image[0];
        context.fillStyle = getColor(color)
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                let index = x + (y * 28)
                if (image[index] != color) {
                    color = image[index]
                    context.fillStyle = getColor(color)
                }
                context.fillRect(x, y, 1, 1)
            }
        }
        currentImageIndicator.innerHTML = `Image #${index}: (${label})\nPredicted: ${data.classification}`
    }
    if (data.trainedEpoch) {
        console.log(`Loss: ${data.loss}`)
        trainButton.textContent = 'Train Epoch'
        trainButton.disabled = false
        nextButton.disabled = false
        previousButton.disabled = false
        drawMode.disabled = false
    }
    if (data.progress) {
        progressBar.textContent = `${Math.round(data.percent * 1000) / 10}%`
        progressBar.value = data.percent
    }
    if (data.batchLoss) {
        let li = document.createElement('li')
        li.style.left = `${pointsPlotted * 5}px`
        li.style.bottom = `${data.percent * 300}px`
        chartList.appendChild(li)
        pointsPlotted += 1
    }
    if (data.accuracy) {
        trainingAccuracy.textContent = `Accuracy on Training Data: ${Math.floor(data.trainingAccuracy * 100)}%`
        testingAccuracy.textContent = `Accuracy on Testing Data: ${Math.floor(data.testingAccuracy * 100)}%`
    }
}
