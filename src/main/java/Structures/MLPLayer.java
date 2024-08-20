package Structures;

import Training.ActivationFunction;
import Training.Environment;

import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;
import java.util.List;

public class MLPLayer extends Layer {
    private static final float CLIP_THRESHOLD = 1.0f;
    private static final float LOSS_SCALE = 128.0f;

    Matrix weights, biases;
    Matrix gradientWeights, gradientBiases;
    ActivationFunction phi;

    public Matrix m;
    public Matrix v;
    public Matrix mBias;
    public Matrix vBias;

    public MLPLayer(int inputSize, int outputSize, ActivationFunction activation, float bias) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new Matrix(outputSize, inputSize);
        float stddev = (float) Math.sqrt(2.0 / (inputSize + outputSize));
        weights.randomize(-stddev, stddev); // Xavier initialization
        biases = new Matrix(outputSize, 1);
        biases.fill(bias);
        gradientWeights = new Matrix(outputSize, inputSize);
        gradientBiases = new Matrix(outputSize, 1);
        phi = activation;

        // Initialize Adam optimiser parameters
        m = new Matrix(outputSize, inputSize);
        v = new Matrix(outputSize, inputSize);
        mBias = new Matrix(outputSize, 1);
        vBias = new Matrix(outputSize, 1);
    }

    @Override
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        if (!(targetLayer instanceof MLPLayer target)) {
            throw new IllegalArgumentException(String.format("Target layer must be a MLPLayer (got: %s)", targetLayer.getClass().getSimpleName()));
        }

        Matrix.copy(this.weights, target.weights);
        Matrix.copy(this.biases, target.biases);
        Matrix.copy(this.gradientWeights, target.gradientWeights);
        Matrix.copy(this.gradientBiases, target.gradientBiases);
        target.phi = this.phi;

        if (ignorePrimitives) return;

        target.inputSize = this.inputSize;
        target.outputSize = this.outputSize;
    }

    @Override
    public MLPLayer copy() {
        MLPLayer copy = new MLPLayer(inputSize, outputSize, phi, 0);
        copyTo(copy, true);
        return copy;
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
                result.set(c, r, phi.activate(result.get(c, r)));
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

        // Apply loss scaling
        matrixGradientOutput = Matrix.multiply(matrixGradientOutput, LOSS_SCALE);

        gradientWeights = Matrix.multiply(matrixGradientOutput, matrixInput.transpose());
        gradientBiases = matrixGradientOutput;

        Matrix gradientInput = Matrix.multiply(weights.transpose(), matrixGradientOutput);

        // Apply activation function derivative
        for (int r = 0; r < gradientInput.rows; r++) {
            for (int c = 0; c < gradientInput.cols; c++) {
                gradientInput.set(c, r, gradientInput.get(c, r) * phi.derivative(matrixInput.get(c, r)));
            }
        }

        return gradientInput;
    }

    @Override
    public void updateParameters(float learningRate) {
        // clip and normalise gradients

        float gradientNorm = (float) Math.sqrt(
                Math.pow(gradientWeights.sumOfSquares(), 2) +
                Math.pow(gradientBiases.sumOfSquares(), 2)
        );

        if (gradientNorm > CLIP_THRESHOLD) {
            float scale = CLIP_THRESHOLD / gradientNorm;
            gradientWeights.multiply(scale);
            gradientBiases.multiply(scale);
        }

        gradientWeights.divide(LOSS_SCALE);
        gradientBiases.divide(LOSS_SCALE);

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

    public Matrix getWeights() {
        return weights;
    }

    public Matrix getGradientWeights() {
        return gradientWeights;
    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getGradientBiases() {
        return gradientBiases;
    }

    public void setWeights(Matrix newWeights) {
        if (newWeights.rows != weights.rows || newWeights.cols != weights.cols) {
            throw new IllegalArgumentException("New weights must have the same dimensions as the current weights.");
        }
        weights = newWeights;
    }

    public void setBiases(Matrix newBiases) {
        if (newBiases.rows != biases.rows || newBiases.cols != biases.cols) {
            throw new IllegalArgumentException("New biases must have the same dimensions as the current biases.");
        }
        biases = newBiases;
    }
}
