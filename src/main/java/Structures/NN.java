package Structures;

import Training.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class NN {
    List<Layer> layers;
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

    protected List<Matrix> forwardPass(Matrix input) {
        List<Matrix> layerOutputs = new ArrayList<>();
        layerOutputs.add(input);

        Matrix currentInput = input;
        for (Layer layer : layers) {
            currentInput = layer.compute(currentInput);
            layerOutputs.add(currentInput);
        }

        return layerOutputs;
    }

    public void backpropagate(Matrix input, Matrix target, List<Matrix> layerOutputs) {
        Matrix output = layerOutputs.getLast();

        Matrix error = Matrix.subtract(target, output);
        Matrix gradientOutput = error; // MSE Loss

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Matrix layerInput = layerOutputs.get(i);

            gradientOutput = currentLayer.backpropagate(layerInput, gradientOutput);
            currentLayer.updateParameters(learningRate);
        }
    }
}
