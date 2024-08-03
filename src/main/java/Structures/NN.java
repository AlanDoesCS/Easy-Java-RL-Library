package Structures;

import Training.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class NN {
    List<Layer> hiddenLayers;
    Layer outputLayer;
    float learningRate; // alpha

    public abstract Matrix getOutput(Matrix input);
    public abstract void saveNN(String filename);
    public abstract void loadNN(String filename);

    private Matrix applyDerivative(Matrix input, ActivationFunction activationFunction) {
        Matrix res = new Matrix(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                res.set(j, i, activationFunction.derivative(input.get(j, i)));
            }
        }
        return res;
    }

    List<Matrix> forwardPass(Matrix input) {    // input is the same as environment state
        List<Matrix> layerInputs = new ArrayList<>();
        layerInputs.add(input); // environment state

        Matrix currentInput = input;
        for (Layer layer : hiddenLayers) {
            currentInput = layer.compute(currentInput);
            layerInputs.add(currentInput);
        }

        return layerInputs;
    }

    public void backpropagate(Matrix input, Matrix target, Matrix output) {
        List<Matrix> layerInputs = forwardPass(input);

        Matrix outErr = Matrix.subtract(target, output);

        // calculate gradient at output layer
        Matrix outGrad = Matrix.elementWiseMultiply(outErr, applyDerivative(output, outputLayer.phi));

        // calculate delta for output layer weights
        Matrix outDelta = Matrix.multiply(outGrad, layerInputs.getLast().transpose());

        // update outputLayer first
        outputLayer.weights.add(Matrix.multiply(outDelta, learningRate));
        outputLayer.biases.add(Matrix.multiply(outGrad, learningRate));

        // go back through hidden layers and update
        Matrix currErr = outErr;
        Matrix currGrad = outGrad;

        for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
            Layer currentLayer = hiddenLayers.get(i);
            Layer nextLayer = (i == hiddenLayers.size() - 1) ? outputLayer : hiddenLayers.get(i + 1);

            // calculate error for the current layer
            currErr = Matrix.multiply(nextLayer.weights.transpose(), currGrad);

            // calculate gradient for current layer
            Matrix layerOutput = currentLayer.compute(layerInputs.get(i));
            currGrad = Matrix.elementWiseMultiply(currErr, applyDerivative(layerOutput, currentLayer.phi));

            // calculate delta for the current layer weights
            Matrix currentDelta = Matrix.multiply(currGrad, layerInputs.get(i).transpose());

            // update current layer weights and biases
            currentLayer.weights.add(Matrix.multiply(currentDelta, learningRate));
            currentLayer.biases.add(Matrix.multiply(currGrad, learningRate));
        }
    }
}
