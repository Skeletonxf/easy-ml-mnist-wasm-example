mod utils;

use wasm_bindgen::prelude::*;

extern crate web_sys;
extern crate easy_ml;
extern crate js_sys;

use easy_ml::matrices::Matrix;
use easy_ml::differentiation::{Record, WengertList};
use easy_ml::linear_algebra;
use easy_ml::numeric::{Numeric};
use easy_ml::numeric::extra::{Real};

use std::convert::TryFrom;
use std::convert::TryInto;
use std::cmp;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// // When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// // allocator.
// #[cfg(feature = "wee_alloc")]
// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
extern {
    fn logProgress(percent: f64);
    fn logBatchLoss(percent: f64);
}

/**
 * Wraps the JavaScript function in a snake_case name
 */
fn log_progress(percent: f64) {
    logProgress(percent);
}

fn log_batch_loss(percent: f64) {
    logBatchLoss(percent);
}

const WIDTH: usize = 28;
const HEIGHT: usize = 28;
const TRAINING_SIZE: usize = 8000;
const TESTING_SIZE: usize = 2000;
const LEARNING_RATE: f64 = 0.2;
const LEARNING_RATE_DISCOUNT_FACTOR: f64 = 0.75;

/// mnist data is grayscale 0-1 range
type Pixel = f64;

#[wasm_bindgen]
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    data: Vec<Pixel>
}

#[wasm_bindgen]
impl Image {
    /// Creates a new Image
    pub fn new() -> Image {
        Image {
            data: Vec::with_capacity(WIDTH * HEIGHT),
        }
    }

    /// Accesses the data buffer of this Image, for JavaScript to fill with the actual data
    pub fn buffer(&mut self) -> *const Pixel {
        self.data.as_ptr()
    }

    pub fn set_length(&mut self) {
        // this is safe because we will only call it after initialising all elements
        // via buffer access on the JS side
        unsafe { self.data.set_len(WIDTH * HEIGHT); }
    }
}

impl From<Image> for Matrix<f64> {
    fn from(image: Image) -> Self {
        Matrix::from_flat_row_major((1,WIDTH * HEIGHT), image.data).map(|pixel| pixel)
    }
}

/// A label type for the MNIST data set, consisting of the 10 digits.
#[wasm_bindgen]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Digit {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
}

impl TryFrom<u8> for Digit {
    type Error = &'static str;

    fn try_from(integer: u8) -> Result<Self, Self::Error> {
        match integer {
            0 => Ok(Digit::Zero),
            1 => Ok(Digit::One),
            2 => Ok(Digit::Two),
            3 => Ok(Digit::Three),
            4 => Ok(Digit::Four),
            5 => Ok(Digit::Five),
            6 => Ok(Digit::Six),
            7 => Ok(Digit::Seven),
            8 => Ok(Digit::Eight),
            9 => Ok(Digit::Nine),
            _ => Err("Number out of range"),
        }
    }
}

impl From<Digit> for u8 {
    fn from(label: Digit) -> Self {
        match label {
            Digit::Zero => 0,
            Digit::One => 1,
            Digit::Two => 2,
            Digit::Three => 3,
            Digit::Four => 4,
            Digit::Five => 5,
            Digit::Six => 6,
            Digit::Seven => 7,
            Digit::Eight => 8,
            Digit::Nine => 9,
        }
    }
}

impl From<Digit> for usize {
    fn from(label: Digit) -> Self {
        match label {
            Digit::Zero => 0,
            Digit::One => 1,
            Digit::Two => 2,
            Digit::Three => 3,
            Digit::Four => 4,
            Digit::Five => 5,
            Digit::Six => 6,
            Digit::Seven => 7,
            Digit::Eight => 8,
            Digit::Nine => 9,
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Dataset {
    images: Vec<Image>,
    labels: Vec<Digit>,
}

#[wasm_bindgen]
impl Dataset {
    pub fn new_training() -> Dataset {
        Dataset {
            images: Vec::with_capacity(TRAINING_SIZE),
            labels: Vec::with_capacity(TRAINING_SIZE),
        }
    }

    pub fn new_testing() -> Dataset {
        Dataset {
            images: Vec::with_capacity(TESTING_SIZE),
            labels: Vec::with_capacity(TESTING_SIZE),
        }
    }

    pub fn add(&mut self, image: Image, label: u8) {
        self.images.push(image);
        self.labels.push(label.try_into().expect("Label invalid"));
    }
}

/// A neural network configuration to classify the Mnist data
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    weights: Vec<Matrix<f64>>,
    epochs: i32,
    //buffer: Vec<f64>,
}

const FIRST_HIDDEN_LAYER_SIZE: usize = 128;
const SECOND_HIDDEN_LAYER_SIZE: usize = 64;

// fn relu<T: Numeric + Copy>(x: T) -> T {
//     if x > T::zero() {
//         x
//     } else {
//         T::zero()
//     }
// }

fn sigmoid<T: Numeric + Real+ Copy>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

#[wasm_bindgen]
impl NeuralNetwork {
    /// Creates a new Neural Network configuration of randomised weights
    /// and a simple feed forward architecture.
    pub fn new() -> NeuralNetwork {
        let mut weights = vec![
            Matrix::empty(0.0, (WIDTH * HEIGHT, FIRST_HIDDEN_LAYER_SIZE)),
            Matrix::empty(0.0, (FIRST_HIDDEN_LAYER_SIZE, SECOND_HIDDEN_LAYER_SIZE)),
            Matrix::empty(0.0, (SECOND_HIDDEN_LAYER_SIZE, 10)),
        ];
        for i in 0..weights.len() {
            for j in 0..weights[i].size().0 {
                for k in 0..weights[i].size().1 {
                    weights[i].set(j, k, (2.0 * js_sys::Math::random()) - 1.0);
                }
            }
        }
        NeuralNetwork {
            weights,
            epochs: 0
            //buffer: Vec::with_capacity(0),
        }
    }

    pub fn layers(&self) -> usize {
        self.weights.len()
    }

    pub fn classify(&self, image: &Image) -> Digit {
        let input: Matrix<f64> = image.clone().into();
        // this neural network is a simple feed forward architecture, so dot product
        // the input through the network weights and apply the sigmoid activation
        // function each step, then take softmax to produce an output
        let output = ((input * &self.weights[0]).map(sigmoid) * &self.weights[1]).map(sigmoid) * &self.weights[2];
        let classification = linear_algebra::softmax(output.row_major_iter());
        // find the index of the largest softmax'd label
        classification.iter()
            // find argmax of the output
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN should not be in list"))
            // convert from usize into a Digit, by construction classiciation only has
            // 10 elements, so the index will fit into a Digit
            .map(|(i, _)| i as u8)
            .unwrap()
            .try_into()
            .unwrap()
    }

    /// Trains the neural net for 1 epoch and returns the average loss on the epoch
    pub fn train(&mut self, training_data: &Dataset) -> f64 {
        log_progress(0.0);
        let history = WengertList::new();
        let mut training = NeuralNetworkTraining::from(&self, &history, self.epochs);
        let loss = training.train_epoch(training_data, &history);
        training.update(self);
        log_progress(1.0);
        self.epochs += 1;
        loss
    }

    /// Computes the accuracy on a dataset and returns the percent correctly classified
    /// as a number between 0 and 1.
    pub fn accuracy(&self, dataset: &Dataset) -> f64 {
        let mut correct = 0;
        for i in 0..dataset.images.len() {
            let prediction = self.classify(&dataset.images[i]);
            if prediction == dataset.labels[i] {
                correct += 1;
            }
        }
        (correct as f64) / (dataset.images.len() as f64)
    }
}

/// At the time of writing, #[wasm_bindgen] does not support lifetimes or type
/// parameters. The Record trait has a lifetime parameter because it must not
/// outlive its WengertList. Unfortunately at the time of writing the WengertList
/// constructor also cannot be a constant function because type parameters other than
/// Sized are not stabalised. Additionally, the WengertList does not implement Sync
/// so it cannot be a shared static variable. The cummulative effect of these restrictions
/// mean that I cannot find a way to pass any structs to JavaScript which include a Record
/// type, even though thread safety is a non concern and any such struct that would be
/// passed to JavaScript would also have been defined to own the WengertList that the Records
/// referenced - ie, such a struct would be completely safe, but I can't find a way to
/// get the Rust type system to agree.
///
/// If you're reading this and #[wasm_bindgen] has added lifetime support, or it's
/// possible to make a WengertList with a &static lifetime, or there's a way to create
/// a struct which owns the WengertList and Records but does not bubble the useless lifetime
/// up then please open an issue or pull request to let me know.
///
/// Until then we will have to not share such types with JavaScript. This is actually
/// not a huge issue, because Records are only needed for training anyway.
#[derive(Clone, Debug)]
struct NeuralNetworkTraining<'a> {
    weights: Vec<Matrix<Record<'a, f64>>>,
    learning_rate: f64,
}

const BATCH_SIZE: usize = 10;

impl <'a> NeuralNetworkTraining<'a> {
    /// Given a WengertList which will be used exclusively for training this struct,
    /// and an existing configuration for weights, creates a new NeuralNetworkTraining
    fn from(configuration: &NeuralNetwork, history: &'a WengertList<f64>, epochs: i32) -> NeuralNetworkTraining<'a> {
        let mut weights = Vec::with_capacity(configuration.weights.len());
        for i in 0..configuration.weights.len() {
            weights.push(Matrix::empty(
                Record::variable(0.0, &history),
                configuration.weights[i].size()
            ));
            for j in 0..configuration.weights[i].size().0 {
                for k in 0..configuration.weights[i].size().1 {
                    let neuron = configuration.weights[i].get(j, k);
                    weights[i].set(j, k, Record::variable(neuron, &history));
                }
            }
        }
        NeuralNetworkTraining {
            weights,
            learning_rate: LEARNING_RATE * LEARNING_RATE_DISCOUNT_FACTOR.powi(epochs)
        }
    }

    /// Updates an existing neural network configuration to the new weights
    /// learned through training.
    fn update(&self, configuration: &mut NeuralNetwork) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].size().0 {
                for k in 0..self.weights[i].size().1 {
                    let neuron = self.weights[i].get(j, k).number;
                    configuration.weights[i].set(j, k, neuron);
                }
            }
        }
    }

    /// Classification is very similar for training, except we stay in floating point
    /// land so we can backprop the error.
    /// This function takes an iterator of Images and Digits, and updates the weights
    /// after getting the errors on the entire batch, returning the average loss for
    /// the batch.
    pub fn train<I>(&mut self, batch: I, learning_rate: f64, history: &'a WengertList<f64>) -> f64
    where I: Iterator<Item = (&'a Image, Digit)> {
        let mut errors = Vec::with_capacity(BATCH_SIZE);
        for (image, label) in batch {
            let input: Matrix<f64> = image.clone().into();
            // this neural network is a simple feed forward architecture, so dot product
            // the input through the network weights and apply the sigmoid activation
            // function each step, then take softmax to produce an output
            let output = {
                let i = input.map(|p| Record::constant(p));
                let layer1 = (i * &self.weights[0]).map(sigmoid);
                let layer2 = (layer1 * &self.weights[1]).map(sigmoid);
                layer2 * &self.weights[2]
            };
            let classification = linear_algebra::softmax(output.row_major_iter());
            //let classification = NeuralNetworkTraining::softmax(output.row_major_iter());
            // Get what we predicted for the true label. To minimise error, we should
            // have predicted 1
            let prediction: Record<f64> = classification[Into::<usize>::into(label)];
            // If we predicted 1 for the true label, error is 0, likewise, if
            // we predicted 0 for the true label, error is 1.
            let error: Record<f64> = Record::constant(1.0) - prediction;
            errors.push(error);
        }
        let batch_size = errors.len();
        let error: Record<f64> = errors.drain(..).sum();
        let derivatives = error.derivatives();
        // update weights to minimise error, note that if error was 0 this
        // trivially does nothing
        self.weights[0].map_mut(|x| x - (derivatives[&x] * learning_rate));
        self.weights[1].map_mut(|x| x - (derivatives[&x] * learning_rate));
        self.weights[2].map_mut(|x| x - (derivatives[&x] * learning_rate));
        // reset gradients
        history.clear();
        self.weights[0].map_mut(Record::do_reset);
        self.weights[1].map_mut(Record::do_reset);
        self.weights[2].map_mut(Record::do_reset);
        error.number / (batch_size as f64)
    }

    /// Performs minibatch SGD for one epoch on all of the training data in a random order,
    /// returning the average loss for the entire epoch.
    pub fn train_epoch(&mut self, training_data: &'a Dataset, history: &'a WengertList<f64>) -> f64 {
        let random_numbers = EndlessRandomGenerator {};
        let random_index_order: Vec<usize> = {
            let mut indexes: Vec<(usize, f64)> = (0..training_data.images.len())
                .zip(random_numbers)
                .collect();
            // sort by the random numbers we zipped
            indexes.sort_by(|(_, i), (_, j)| i.partial_cmp(j).unwrap());
            // drop the random numbers in the now randomised list of indexes
            indexes.drain(..).map(|(x, _)| x).collect()
        };
        let mut epoch_losses = 0.0;
        let mut batch_losses = 0.0;
        let mut progress = 0;
        let mut i = 0;
        loop {
            // compute the start and end indexes which will slice the random_index_order vec
            // to obtain a slice of indexes into the training data. Until reaching the end
            // of the datsset this will always be BATCH_SIZE, but may be smaller on the final
            // one.
            let start = i;
            let end = cmp::min(random_index_order.len(), start + BATCH_SIZE);
            let batch_indexes = &random_index_order[start..end];
            if progress % 10 == 0 {
                log_progress(i as f64 / (training_data.images.len() as f64));
            }
            // create a batch of tuples of referenced images and corresponding labels
            let batch = batch_indexes.iter()
                .map(|&index| (&training_data.images[index], training_data.labels[index]));
            let loss = self.train(batch, self.learning_rate, history);
            epoch_losses += loss;
            batch_losses += loss;
            // Report progress to the Web Worker after every 100 images (10 batches
            // for a BATCH_SIZE of 10).
            if progress % 10 == 0 && progress != 0 {
                log_batch_loss(batch_losses / 10.0);
                batch_losses = 0.0;
            }
            progress += 1;
            if end == random_index_order.len() {
                break;
            }
            i += BATCH_SIZE;
        }
        epoch_losses / (training_data.images.len() as f64)
    }
}

#[wasm_bindgen]
pub fn prepare() {
    utils::set_panic_hook();
}

struct EndlessRandomGenerator {}

impl Iterator for EndlessRandomGenerator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        // always return Some, hence this iterator is infinite
        Some(js_sys::Math::random())
    }
}
