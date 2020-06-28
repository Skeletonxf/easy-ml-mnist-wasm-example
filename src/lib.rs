mod utils;

use wasm_bindgen::prelude::*;

extern crate web_sys;
extern crate easy_ml;

use easy_ml::matrices::Matrix;
use easy_ml::differentiation::{Record, WengertList};

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
type Pixel = u8; // mnist data is grayscale 0-255

#[wasm_bindgen]
pub fn train() {
    utils::set_panic_hook();
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
