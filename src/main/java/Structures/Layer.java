package Structures;

import Training.ActivationFunction;

public class Layer {
    Matrix weights, biases;
    ActivationFunction phi;
    private final int inputSize, outputSize;

    public Layer(int inputSize, int outputSize, ActivationFunction activation, float bias) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new Matrix(outputSize, inputSize);
        weights.randomize();
        biases = new Matrix(outputSize, 1);
        biases.fill(bias);
        phi = activation;
    }

    public Matrix compute(Matrix input) {
        return Matrix.multiply(weights, input, 16);
    }

    public int getOutputSize() {
        return outputSize;
    }
    public int getInputSize() {
        return inputSize;
    }
}
