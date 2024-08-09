package Structures;

import Training.ActivationFunction;
import Training.Environment;

import java.util.ArrayList;
import java.util.List;

public class MLPLayer extends Layer {
    Matrix weights, biases;
    private Matrix gradientWeights, gradientBiases;
    ActivationFunction phi;

    public MLPLayer(int inputSize, int outputSize, ActivationFunction activation, float bias) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new Matrix(outputSize, inputSize);
        weights.randomize();
        biases = new Matrix(outputSize, 1);
        biases.fill(bias);
        gradientWeights = new Matrix(outputSize, inputSize);
        gradientBiases = new Matrix(outputSize, 1);
        phi = activation;
    }

    @Override
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

    @Override
    public Matrix backpropagate(Matrix input, Matrix gradientOutput) {
        gradientWeights = Matrix.multiply(gradientOutput, input.transpose());
        gradientBiases = gradientOutput;

        Matrix gradientInput = Matrix.multiply(weights.transpose(), gradientOutput);

        // Apply activation function derivative
        for (int i = 0; i < gradientInput.rows; i++) {
            for (int j = 0; j < gradientInput.cols; j++) {
                gradientInput.set(j, i, gradientInput.get(j, i) * phi.derivative(input.get(j, i)));
            }
        }

        return gradientInput;
    }

    @Override
    public void updateParameters(float learningRate) {
        weights.subtract(Matrix.multiply(gradientWeights, learningRate));
        biases.subtract(Matrix.multiply(gradientBiases, learningRate));
    }

    @Override
    public String toString() {
        return "MLPLayer: in:" + inputSize + "\tout:" + outputSize + "\tactivation:" + phi;
    }

    /*
    ---------------------------------------------------------

    STATIC METHODS

    ---------------------------------------------------------
     */
    public static List<MLPLayer> createMLPLayers(int inputSize, List<Integer> sizes, List<ActivationFunction> activationFunctions, List<Integer> biases) {
        if (sizes.size() != activationFunctions.size() || sizes.size() != biases.size() || activationFunctions.size() != biases.size()) {
            throw new IllegalArgumentException("Lists must have the same size: "+sizes.size()+", "+activationFunctions.size()+", "+biases.size());
        }

        List<MLPLayer> layers = new ArrayList<>();

        int currInputSize = inputSize;

        for (int i=0; i<sizes.size(); i++) {
            layers.add(new MLPLayer(currInputSize, sizes.get(i), activationFunctions.get(i), biases.get(i)));
            currInputSize = sizes.get(i);   // input of next layer is size of previous layer
        }

        return layers;
    }
}
