package Structures;

import Training.ActivationFunction;

import java.util.List;

public abstract class NN {
    List<Layer> hiddenLayers;
    Layer outputLayer;
    float learningRate; // alpha

    public abstract Matrix getOutput(Matrix input);

    private Matrix applyDerivative(Matrix input, ActivationFunction activationFunction) {
        Matrix res = new Matrix(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                res.set(i, j, activationFunction.derivative(input.get(i, j)));
            }
        }
        return res;
    }

    public void backpropagate(Matrix input, Matrix target, Matrix output) {
        Matrix outErr = Matrix.subtract(target, output);

        // calculate gradient at output layer
        Matrix outGrad = applyDerivative(output, outputLayer.phi);
        outGrad = Matrix.multiply(Matrix.multiply(outGrad, outErr), learningRate);

        // calculate delta for output layer weights
        Matrix outDelta = Matrix.multiply(outGrad, hiddenLayers.getLast().compute(input));

        // update outputLayer first
        outputLayer.weights.add(outDelta);
        outputLayer.biases.add(outGrad);

        // go back through hidden layers and update
        Matrix currErr = outErr;
        for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
            Layer currentLayer = hiddenLayers.get(i);
            Layer nextLayer = (i == hiddenLayers.size() - 1) ? outputLayer : hiddenLayers.get(i + 1);

            // calculate error for the current layer
            currErr = Matrix.multiply(nextLayer.weights, currErr);

            // calculate gradient for current layer
            Matrix currentGradient = applyDerivative(currentLayer.compute(input), currentLayer.phi);
            currentGradient.multiply(currErr);
            currentGradient.multiply(learningRate);

            // calculate delta for the current layer weights
            Matrix currentInput = (i == 0) ? input : hiddenLayers.get(i - 1).compute(input);
            Matrix currentDelta = Matrix.multiply(currentGradient, currentInput);

            // update current layer weights and biases
            currentLayer.weights.add(currentDelta);
            currentLayer.biases.add(currentGradient);
        }
    }
}
