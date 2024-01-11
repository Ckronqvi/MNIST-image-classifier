# MNIST image classifier

<div style="width: 80%; margin: 0 auto;">

## Purpose

> I wanted to try and tackle the MNIST handwritten image challenge using
> a simple convolutional neural network. The library I used was the
> [tch](https://github.com/LaurentMazare/tch-rs), which provides
> wrappers around the C++ PyTorch api.

 

## Results

> I was able to achieve over 99% training accuracy and over 98.5% test
> accuracy. With sufficient training and adjustments to the network, I
> could probably achieve a testing accuracy of 99% as well, but I am
> satisfied with the current results.

![Training graph](/CNN/accuracy.png)

## How to run
### Requirements:

 - [LibTorch](https://pytorch.org/cppdocs/installing.html)
 - [Rust](https://www.rust-lang.org/tools/install)
 - [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
### Running:
Put the MNIST dataset into the */data* folder at the root of the project. After that, you can modify the Config.toml file to adjust the settings of the network for your liking and then simply run `cargo build` and `cargo run`
</div>
