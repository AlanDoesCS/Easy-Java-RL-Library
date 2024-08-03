package Structures;

import Training.ActivationFunction;
import Training.Environment;

import java.util.ArrayList;
import java.util.List;

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

    public String toString() {
        return "in:" + inputSize + "\tout:" + outputSize + "\tactivation:" + phi;
    }

    public int getOutputSize() {
        return outputSize;
    }
    public int getInputSize() {
        return inputSize;
    }

    /*
    ---------------------------------------------------------

    STATIC METHODS

    ---------------------------------------------------------
     */
    public static List<Layer> createHiddenLayers(List<Integer> sizes, List<ActivationFunction> activationFunctions, List<Integer> biases) {
        if (sizes.size() != activationFunctions.size() || sizes.size() != biases.size() || activationFunctions.size() != biases.size()) {
            throw new IllegalArgumentException("Lists must have the same size: "+sizes.size()+", "+activationFunctions.size()+", "+biases.size());
        }

        List<Layer> layers = new ArrayList<>();

        int currInputSize = Environment.getStateSpace();    // input layer is always environment state

        for (int i=0; i<sizes.size(); i++) {
            layers.add(new Layer(currInputSize, sizes.get(i), activationFunctions.get(i), biases.get(i)));
            currInputSize = sizes.get(i);   // input of next layer is size of previous layer
        }

        return layers;
    }

    public static String toString(List<Layer> layers) {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i<layers.size(); i++) {
            sb.append("Layer ").append(i).append(": ").append(layers.get(i).toString()).append("\n");
        }
        return sb.toString();
    }
}
