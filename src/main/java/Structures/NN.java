package Structures;

import Training.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class NN {
    List<Layer> layers;
    float learningRate; // alpha

    public abstract Object getOutput(Object input);
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

    protected List<Object> forwardPass(Object input) {
        List<Object> layerOutputs = new ArrayList<>();
        layerOutputs.add(input);

        Object currentInput = input;
        int i=0;
        for (Layer layer : layers) {
            currentInput = layer.compute(currentInput);
            layerOutputs.add(currentInput);
        }

        return layerOutputs;
    }

    public void backpropagate(Object input, Matrix target, List<Object> layerOutputs) {
        if (!(target instanceof Matrix)) {
            throw new IllegalArgumentException("Target must be a Matrix.");
        }

        Matrix output = (Matrix) layerOutputs.getLast();
        Matrix error = Matrix.subtract(target, output);
        Object gradientOutput = error; // Assuming MSE Loss, use the error directly as the gradient.

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Object layerInput = layerOutputs.get(i);

            gradientOutput = currentLayer.backpropagate(layerInput, gradientOutput);

            currentLayer.updateParameters(learningRate);
        }
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
}
