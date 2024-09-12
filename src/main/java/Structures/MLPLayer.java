package Structures;

import Tools.math;
import Training.ActivationFunctions.ActivationFunction;

/**
 * Represents a Multi-Layer Perceptron (MLP) layer in a neural network.
 * <p>
 * This class extends the abstract Layer class and provides functionality
 * for forward and backward propagation, parameter updates, and other
 * operations specific to MLP layers.
 * </p>
 */
public class MLPLayer extends Layer {
    private static final double CLIP_THRESHOLD = 1.0f; // Threshold for gradient clipping
    private static final double LOSS_SCALE = 128.0f;

    MatrixDouble weights, biases;
    MatrixDouble gradientWeights, gradientBiases;
    ActivationFunction phi;
    double lambda;

    public MatrixDouble m;
    public MatrixDouble v;
    public MatrixDouble mBias;
    public MatrixDouble vBias;

    public MLPLayer(int inputSize, int outputSize, ActivationFunction activation, double bias, double lambda) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new MatrixDouble(outputSize, inputSize);
        double stddev = Math.sqrt(2.0 / (inputSize)); // He initialization
        weights.randomize(-stddev, stddev);
        biases = new MatrixDouble(outputSize, 1);
        biases.fill(bias);
        gradientWeights = new MatrixDouble(outputSize, inputSize);
        gradientBiases = new MatrixDouble(outputSize, 1);
        phi = activation;
        this.lambda = lambda;

        // Initialize Adam optimizer parameters
        m = new MatrixDouble(outputSize, inputSize);
        v = new MatrixDouble(outputSize, inputSize);
        mBias = new MatrixDouble(outputSize, 1);
        vBias = new MatrixDouble(outputSize, 1);
    }

    @Override
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        targetLayer.alpha = this.alpha;

        if (!(targetLayer instanceof MLPLayer target)) {
            throw new IllegalArgumentException(String.format("Target layer must be a MLPLayer (got: %s)", targetLayer.getClass().getSimpleName()));
        }

        MatrixDouble.copy(this.weights, target.weights);
        MatrixDouble.copy(this.biases, target.biases);
        MatrixDouble.copy(this.gradientWeights, target.gradientWeights);
        MatrixDouble.copy(this.gradientBiases, target.gradientBiases);
        target.phi = this.phi;

        if (ignorePrimitives) return;

        target.inputSize = this.inputSize;
        target.outputSize = this.outputSize;
        target.lambda = this.lambda;
    }

    @Override
    public MLPLayer copy() {
        MLPLayer copy = new MLPLayer(inputSize, outputSize, phi, 0, lambda);
        copyTo(copy, true);
        return copy;
    }

    @Override
    public Object compute(Object input) {
        if (!(input instanceof MatrixDouble matrixInput)) {
            throw new IllegalArgumentException("Expected input to be a MatrixDouble.");
        }

        MatrixDouble result = MatrixDouble.multiply(weights, matrixInput);
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
    public MatrixDouble backpropagate(Object input, Object gradientOutput) {
        if (!(input instanceof MatrixDouble matrixInput)) {
            throw new IllegalArgumentException("Expected input to be a MatrixDouble.");
        }
        if (!(gradientOutput instanceof MatrixDouble matrixGradientOutput)) {
            throw new IllegalArgumentException("Expected gradientOutput to be a MatrixDouble.");
        }

        // Apply loss scaling
        matrixGradientOutput = MatrixDouble.multiply(matrixGradientOutput, LOSS_SCALE);

        this.gradientWeights = MatrixDouble.multiply(matrixGradientOutput, matrixInput.transpose());
        this.gradientBiases = matrixGradientOutput;

        MatrixDouble gradientInput = MatrixDouble.multiply(weights.transpose(), matrixGradientOutput);

        // Apply activation function derivative
        for (int r = 0; r < gradientInput.rows; r++) {
            for (int c = 0; c < gradientInput.cols; c++) {
                gradientInput.set(c, r, gradientInput.get(c, r) * phi.derivative(matrixInput.get(c, r)));
            }
        }

        return gradientInput;
    }

    @Override
    public void updateParameters(double learningRate) {
        // Calculate gradient norms for clipping
        double gradientNorm = Math.sqrt(
                gradientWeights.sumOfSquares() + gradientBiases.sumOfSquares()
        );

        // Clip gradients if they exceed the threshold
        if (gradientNorm > CLIP_THRESHOLD) {
            double scale = CLIP_THRESHOLD / gradientNorm;
            gradientWeights.multiply(scale);
            gradientBiases.multiply(scale);
        }

        // Normalize gradients
        gradientWeights.divide(LOSS_SCALE);
        gradientBiases.divide(LOSS_SCALE);

        // L2 regularization (weight decay)
        weights.multiply(1.0f - learningRate * lambda);
        biases.multiply(1.0f - learningRate * lambda);

        // Update weights and biases using the scaled gradients
        weights.subtract(MatrixDouble.multiply(gradientWeights, learningRate));
        biases.subtract(MatrixDouble.multiply(gradientBiases, learningRate));
    }

    @Override
    public String toString() {
        return "MLPLayer: in:" + inputSize + "\tout:" + outputSize + "\tactivation:" + phi;
    }

    public MatrixDouble getWeights() {
        return weights;
    }

    public MatrixDouble getGradientWeights() {
        return gradientWeights;
    }

    public MatrixDouble getBiases() {
        return biases;
    }

    public MatrixDouble getGradientBiases() {
        return gradientBiases;
    }

    public void setWeights(MatrixDouble newWeights) {
        if (newWeights.rows != weights.rows || newWeights.cols != weights.cols) {
            throw new IllegalArgumentException("New weights must have the same dimensions as the current weights.");
        }
        weights = newWeights;
    }

    public void setBiases(MatrixDouble newBiases) {
        if (newBiases.rows != biases.rows || newBiases.cols != biases.cols) {
            throw new IllegalArgumentException("New biases must have the same dimensions as the current biases.");
        }
        biases = newBiases;
    }

    @Override
    public void dumpInfo() {
        System.out.println("MLP Layer | Input Size: "+inputSize+" | Output Size: "+outputSize);
        System.out.println("- weight average: "+weights.getMeanAverage()+", range: "+ math.min(weights)+", "+math.max(weights));
        System.out.println("- gradient weight average: "+gradientWeights.getMeanAverage()+", range: "+ math.min(gradientWeights)+", "+math.max(gradientWeights));
        System.out.println("- biases: " + biases.toRowMatrix());
        System.out.println("- gradient biases: " + gradientBiases.toRowMatrix());
    }
}
