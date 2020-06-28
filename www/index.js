import { train } from 'mnist'

let trainButton = document.querySelector('#train')
trainButton.addEventListener('click', () => {
  train()
})
