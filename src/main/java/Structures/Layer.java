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
        Matrix result = Matrix.multiply(weights, input);
        result.add(biases);
        // Apply activation function
        for (int r = 0; r < result.rows; r++) {
            for (int c = 0; c < result.cols; c++) {
                result.set(c, r, phi.activate(result.get(c, r)));
            }
        }
        return result;
    }

    public int getOutputSize() {
        return outputSize;
    }
    public int getInputSize() {
        return inputSize;
    }
}
