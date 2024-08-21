package Structures;

import Training.ActivationFunctions.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class NN {
    List<Layer> layers;
    double learningRate; // alpha

    public abstract Object getOutput(Object input);
    public abstract void saveNN(String filename);
    public abstract void loadNN(String filename);

    private MatrixDouble applyDerivative(MatrixDouble input, ActivationFunction activationFunction) {
        MatrixDouble res = new MatrixDouble(input.rows, input.cols);
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
            System.out.println("Layer "+i);

            currentInput = layer.compute(currentInput);
            layerOutputs.add(currentInput);
            i++;
        }

        return layerOutputs;
    }

    public void backpropagate(Object input, MatrixDouble target, List<Object> layerOutputs) {
        if (!(target instanceof MatrixDouble)) {
            throw new IllegalArgumentException("Target must be a MatrixDouble.");
        }

        MatrixDouble output = (MatrixDouble) layerOutputs.getLast();
        MatrixDouble error = MatrixDouble.subtract(target, output);
        Object gradientOutput = error; // Assuming MSE Loss, use the error directly as the gradient.

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Object layerInput = layerOutputs.get(i);

            gradientOutput = currentLayer.backpropagate(layerInput, gradientOutput);

            currentLayer.updateParameters(learningRate);
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
