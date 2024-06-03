package Structures;

import Training.ActivationFunction;

public class Layer {
    Matrix weights, biases;
    ActivationFunction phi;

    public Layer(int inputSize, int outputSize, ActivationFunction activation, float bias) {
        weights = new Matrix(outputSize, inputSize);
        weights.randomize();
        biases = new Matrix(outputSize, 1);
        biases.fill(bias);
        phi = activation;
    }

    public Matrix compute(Matrix input) {
        Matrix result = Matrix.multiply(weights, input);

        System.out.println(result);
        return result;
    }
}
