import numpy as np
np.random.seed(42)

class Layer:
    #Each layer is capable of doing 2 things
    # - > Forward pass : Process Input to get Output
    # - > Backward pass : Propagate gradients throught itself
    # Some Layer have learnable paameters wich they update during backward pass
    def __init__(self):
        pass

    def forward(self, input):
        #Take input data shape [batch, input_units], returns outputs data [batch, output_units]
        return input

    def backward(self, input, grad_output):
        #Perform a backpropagation step throught the layer, with respect given to input


        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


class ReLU(Layer):
    def __init__(self):
        #ReLU layer simplu applies elementwise recifified linear unit to all inputs
        pass
    def forward(self, input):
        #Apply eleemntwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t ReLU input
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):

        super().__init__()
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def forward(self, input):

        return np.dot(input, self.weights) + self.biases