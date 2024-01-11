extern crate tch;

use tch::{nn, Tensor, Kind};


#[derive(Debug)]
/// Represents a neural network with convolutional and fully connected layers.
pub struct Network {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,  
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl nn::ModuleT for Network {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)
            .log_softmax(-1, Kind::Float)
    }
}

impl Network {
    pub fn new(vs: &nn::Path) -> Network {
        Network {
             conv1: nn::conv2d(vs, 1, 32, 5, Default::default()),
             conv2: nn::conv2d(vs, 32, 64, 5, Default::default()),
             fc1: nn::linear(vs, 1024, 1024, Default::default()),
             fc2: nn::linear(vs, 1024, 10, Default::default()),
        }
    }
}