# Easy ML MNIST Web Assembly Example

Simple MNIST Neural Network scaffold for demonstrating Rust code in the browser.

Uses `wasm-pack` to build the web assembly. The webpage can be accessed by
running the included Node.js server.

## About

This project is a template for doing machine learning in the browser via Rust
code loaded as WebAssembly. The code trains a simple feedforward neural network
on a subset of the MNIST data using [Easy ML](https://crates.io/crates/easy-ml)
with mini batching and automatic differentiation.

## Screenshots

<img src="../master/screenshots/webpage.png?raw=true" height="250px"></img>

## Limitations

At the time of writing,
[there is not widespread support](https://caniuse.com/#feat=mdn-javascript_statements_import_worker_support)
for ES6 module imports in Web Workers. Hence, this scaffold uses
`importScripts` to import the web assembly in the web worker, and
`wasm-pack build --target no-modules --out-dir www/pkg` to generate the web
assembly and JavaScript code for the Web Worker to import.

This makes the code a little less nice than if we could use ES6 modules
everywhere, but is worth it as training a machine learning system in the
main loop will almost certainly freeze up the browser or web page.

If you're reading this in the future, and ES6 imports are widely available in
Web Workers then please open an issue so I can update the template to use
module imports.

## Running

First generate the web assembly by running
`wasm-pack build --target no-modules --out-dir www/pkg`. Then `cd www` and
run `npm install` to install the server, then `npm run start` to launch it.

Open `http://localhost:8080/`

The webpack dev server will automatically reload JavaScript when you edit it,
but the Rust code must be manually rebuilt with
`wasm-pack build --target no-modules --out-dir www/pkg` ran from the root of the project directory each time you make
changes.

## Background info

For further information on Rust and WebAssembly checkout the tutorials by the rust-wasm group.
- https://rustwasm.github.io/docs/book/introduction.html
- https://rustwasm.github.io/docs/wasm-pack/introduction.html
