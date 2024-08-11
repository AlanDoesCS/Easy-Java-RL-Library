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
    public Object compute(Object input) {
        if (!(input instanceof Matrix matrixInput)) {
            throw new IllegalArgumentException("Expected input to be a Matrix.");
        }

        Matrix result = Matrix.multiply(weights, matrixInput);
        result.add(biases);

        // Apply activation function
        for (int r = 0; r < result.rows; r++) {
            for (int c = 0; c < result.cols; c++) {
                result.set(r, c, phi.activate(result.get(r, c)));
            }
        }
        return result;
    }

    @Override
    public Matrix backpropagate(Object input, Object gradientOutput) {
        if (!(input instanceof Matrix)) {
            throw new IllegalArgumentException("Expected input to be a Matrix.");
        }
        if (!(gradientOutput instanceof Matrix)) {
            throw new IllegalArgumentException("Expected gradientOutput to be a Matrix.");
        }

        Matrix matrixInput = (Matrix) input;
        Matrix matrixGradientOutput = (Matrix) gradientOutput;

        gradientWeights = Matrix.multiply(matrixGradientOutput, matrixInput.transpose());
        gradientBiases = matrixGradientOutput;

        Matrix gradientInput = Matrix.multiply(weights.transpose(), matrixGradientOutput);

        // Apply activation function derivative
        for (int i = 0; i < gradientInput.rows; i++) {
            for (int j = 0; j < gradientInput.cols; j++) {
                gradientInput.set(i, j, gradientInput.get(i, j) * phi.derivative(matrixInput.get(i, j)));
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
