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
        for (int r = 0; r < input.rows; r++) {
            for (int c = 0; c < input.cols; c++) {
                res.set(c, r, activationFunction.derivative(input.get(c, r)));
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
            // System.out.println("Layer "+i);

            currentInput = layer.compute(currentInput);
            layerOutputs.add(currentInput);
            i++;
        }

        return layerOutputs;
    }

    public abstract void backpropagate(Object input, MatrixDouble target, List<Object> layerOutputs);

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
