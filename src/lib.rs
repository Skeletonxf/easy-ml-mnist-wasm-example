mod utils;

use wasm_bindgen::prelude::*;

extern crate web_sys;
extern crate easy_ml;

use easy_ml::matrices::Matrix;
use easy_ml::differentiation::{Record, WengertList};

use std::convert::TryFrom;
use std::convert::TryInto;

// A macro to provide `println!(..)`-style syntax for `console.log` logging.
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    alert("Hello, mnist!");
}

const WIDTH: usize = 28;
const HEIGHT: usize = 28;
const TRAINING_SIZE: usize = 8000;
const TESTING_SIZE: usize = 2000;
/// mnist data is grayscale 0-255

type Pixel = u8;

#[wasm_bindgen]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    data: Vec<Pixel>
}

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

#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Dataset {
    images: Vec<Image>,
    labels: Vec<Digit>,
}

impl Dataset {
    pub fn new() -> Dataset {
        Dataset {
            images: Vec::with_capacity(TRAINING_SIZE),
            labels: Vec::with_capacity(TRAINING_SIZE),
        }
    }

    pub fn add(&mut self, image: Image, label: u8) {
        self.images.push(image);
        self.labels.push(label.try_into().expect("Label invalid"));
    }
}

#[wasm_bindgen]
pub fn prepare() {
    utils::set_panic_hook();
}

#[wasm_bindgen]
pub fn train() {
    let history = WengertList::new();
    let weights = vec![
        Matrix::empty(Record::variable(0.0, &history), (WIDTH * HEIGHT, 50))
    ];
    let data = vec![
        Matrix::<Pixel>::from(vec![
            vec![ 0, 255, 0 ],
            vec![ 0, 255, 0 ],
            vec![ 0, 255, 0 ]])
    ];
    log!("Starting training");
    log!("Done");
}
