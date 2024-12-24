package Structures;

import Tools.math;
import Training.ActivationFunctions.ActivationFunction;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Represents a convolutional layer in a neural network.
 * <p>
 * This class extends the abstract Layer class and provides
 * functionality for forward and backward propagation, parameter
 * updates, and other operations specific to convolutional layers.
 * </p>
 */
public class ConvLayer extends Layer {
    private static final int PARALLELISM_THRESHOLD = 32; // Threshold for parallelizing loops
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

    // Parameters
    public double[][][][] filters; // [numFilters][depth][height][width]
    public double[] biases; // [numFilters]

    // Gradients
    private double[][][][] gradientFilters;
    private double[] gradientBiases;

    // Adam optimizer parameters
    public double[][][][] mFilters, vFilters;
    public double[] mBiases, vBiases;

    private int strideX, strideY;
    private int paddingX, paddingY;
    public int filterSize; // Assumes square filters
    private int numFilters;
    private int inputWidth, inputHeight, inputDepth;
    private int outputWidth, outputHeight;

    private ActivationFunction activationFunction;

    public ConvLayer(ActivationFunction activationFunction, int inputWidth, int inputHeight, int inputDepth,
                     int filterSize, int numFilters, int strideX, int strideY, int paddingX, int paddingY, String... args) {
        this.activationFunction = activationFunction;
        this.inputSize = inputWidth * inputHeight * inputDepth;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.strideX = strideX;
        this.strideY = strideY;
        this.paddingX = paddingX;
        this.paddingY = paddingY;

        this.outputWidth = (inputWidth - filterSize + 2 * paddingX) / strideX + 1;
        this.outputHeight = (inputHeight - filterSize + 2 * paddingY) / strideY + 1;
        this.outputSize = outputWidth * outputHeight * numFilters;

        // Initialize filters and biases
        filters = new double[numFilters][inputDepth][filterSize][filterSize];
        biases = new double[numFilters];
        gradientFilters = new double[numFilters][inputDepth][filterSize][filterSize];
        gradientBiases = new double[numFilters];

        if (args.length == 0 || !Arrays.asList(args).contains("noInit")) {
            initializeParameters();
        }

        // Initialize Adam optimizer parameters
        mFilters = new double[numFilters][inputDepth][filterSize][filterSize];
        vFilters = new double[numFilters][inputDepth][filterSize][filterSize];
        mBiases = new double[numFilters];
        vBiases = new double[numFilters];
    }

    private void initializeParameters() {
        // He initialization (for ReLU)
        double stdDev = Math.sqrt(2.0 / (inputDepth * filterSize * filterSize));
        for (int i = 0; i < filters.length; i++) {
            for (int j = 0; j < filters[i].length; j++) {
                for (int k = 0; k < filters[i][j].length; k++) {
                    for (int l = 0; l < filters[i][j][k].length; l++) {
                        filters[i][j][k][l] = math.randomDouble(-stdDev, stdDev);
                    }
                }
            }
        }

        Arrays.fill(biases, 0);
    }

    @Override
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        targetLayer.alpha = this.alpha;

        if (!(targetLayer instanceof ConvLayer)) {
            throw new IllegalArgumentException(String.format("Target layer must be a ConvLayer (got: %s)", targetLayer.getClass().getSimpleName()));
        }
        ConvLayer target = (ConvLayer) targetLayer;

        // Copy filters and biases
        for (int f = 0; f < this.filters.length; f++) {
            for (int d = 0; d < this.filters[f].length; d++) {
                for (int h = 0; h < this.filters[f][d].length; h++) {
                    System.arraycopy(this.filters[f][d][h], 0, target.filters[f][d][h], 0, this.filters[f][d][h].length);
                }
            }
        }
        System.arraycopy(this.biases, 0, target.biases, 0, this.biases.length);

        // Copy gradients
        for (int f = 0; f < this.gradientFilters.length; f++) {
            for (int d = 0; d < this.gradientFilters[f].length; d++) {
                for (int h = 0; h < this.gradientFilters[f][d].length; h++) {
                    System.arraycopy(this.gradientFilters[f][d][h], 0, target.gradientFilters[f][d][h], 0, this.gradientFilters[f][d][h].length);
                }
            }
        }
        System.arraycopy(this.gradientBiases, 0, target.gradientBiases, 0, this.gradientBiases.length);

        // Copy moment estimates
        for (int f = 0; f < this.mFilters.length; f++) {
            for (int d = 0; d < this.mFilters[f].length; d++) {
                for (int h = 0; h < this.mFilters[f][d].length; h++) {
                    System.arraycopy(this.mFilters[f][d][h], 0, target.mFilters[f][d][h], 0, this.mFilters[f][d][h].length);
                    System.arraycopy(this.vFilters[f][d][h], 0, target.vFilters[f][d][h], 0, this.vFilters[f][d][h].length);
                }
            }
        }
        System.arraycopy(this.mBiases, 0, target.mBiases, 0, this.mBiases.length);
        System.arraycopy(this.vBiases, 0, target.vBiases, 0, this.vBiases.length);

        target.activationFunction = this.activationFunction;

        if (ignorePrimitives) return;

        target.strideX = this.strideX;
        target.strideY = this.strideY;
        target.paddingX = this.paddingX;
        target.paddingY = this.paddingY;
        target.filterSize = this.filterSize;
        target.numFilters = this.numFilters;
        target.inputWidth = this.inputWidth;
        target.inputHeight = this.inputHeight;
        target.inputDepth = this.inputDepth;
        target.outputWidth = this.outputWidth;
        target.outputHeight = this.outputHeight;
    }

    @Override
    public ConvLayer copy() {
        ConvLayer copyLayer = new ConvLayer(activationFunction, inputWidth, inputHeight, inputDepth, filterSize,
                numFilters, strideX, strideY, paddingX, paddingY, "noInit");
        copyTo(copyLayer, true);
        return copyLayer;
    }

    public int getNumFilters() { return numFilters; }
    public int getOutputDepth() {
        return numFilters;
    }
    public int getOutputHeight() {
        return outputHeight;
    }
    public int getOutputWidth() {
        return outputWidth;
    }

    public void dumpFilters() {
        for (int f = 0; f < numFilters; f++) {
            System.out.println("Filter " + f + ":");
            for (int d = 0; d < inputDepth; d++) {
                for (int h = 0; h < filterSize; h++) {
                    for (int w = 0; w < filterSize; w++) {
                        System.out.print(filters[f][d][h][w] + " ");
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println("Bias: " + biases[f]);
        }
    }

    public double getFilterMin() {
        double min = Float.MAX_VALUE;
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int h = 0; h < filterSize; h++) {
                    for (int w = 0; w < filterSize; w++) {
                        if (filters[f][d][h][w] < min) {
                            min = filters[f][d][h][w];
                        }
                    }
                }
            }
        }
        return min;
    }
    public double getFilterMax() {
        double max = Double.MIN_VALUE;
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int h = 0; h < filterSize; h++) {
                    for (int w = 0; w < filterSize; w++) {
                        if (filters[f][d][h][w] > max) {
                            max = filters[f][d][h][w];
                        }
                    }
                }
            }
        }
        return max;
    }
    public Double getAverageFilterValue() {
        double sum = 0;
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int h = 0; h < filterSize; h++) {
                    for (int w = 0; w < filterSize; w++) {
                        sum += filters[f][d][h][w];
                    }
                }
            }
        }
        return sum / (numFilters * inputDepth * filterSize * filterSize);
    }

    public int getInputDepth() {
        return inputDepth;
    }

    public int getFilterSize() {
        return filterSize;
    }

    public double[][][][] getGradientFilters() {
        return gradientFilters;
    }

    public double[] getGradientBiases() {
        return gradientBiases;
    }

    @Override
    public Object compute(Object input) {
        if (!(input instanceof Tensor)) {
            throw new IllegalArgumentException("Expected input to be a Tensor.");
        }
        Tensor tensorInput = (Tensor) input;

        if (tensorInput.getDepth() != inputDepth || tensorInput.getHeight() != inputHeight || tensorInput.getWidth() != inputWidth) {
            throw new IllegalArgumentException("Input dimensions do not match expected dimensions.");
        }

        double[][][] outputData = new double[numFilters][outputHeight][outputWidth];
        POOL.invoke(new ComputeTask(tensorInput, outputData, 0, numFilters));
        return new Tensor(outputData);
    }

    private class ComputeTask extends RecursiveAction {
        private final Tensor input;
        private final double[][][] output;
        private final int startFilter, endFilter;

        ComputeTask(Tensor input, double[][][] output, int startFilter, int endFilter) {
            this.input = input;
            this.output = output;
            this.startFilter = startFilter;
            this.endFilter = endFilter;
        }

        @Override
        protected void compute() {
            if (endFilter - startFilter <= PARALLELISM_THRESHOLD) {
                computeSequential();
            } else {
                int midFilter = (startFilter + endFilter) / 2;
                invokeAll(
                        new ComputeTask(input, output, startFilter, midFilter),
                        new ComputeTask(input, output, midFilter, endFilter)
                );
            }
        }

        private void computeSequential() {
            for (int f = startFilter; f < endFilter; f++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        double sum = 0;
                        for (int d = 0; d < inputDepth; d++) {
                            for (int k = 0; k < filterSize; k++) {
                                for (int l = 0; l < filterSize; l++) {
                                    int inputI = i * strideY - paddingY + k;
                                    int inputJ = j * strideX - paddingX + l;
                                    if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                        sum += input.get(d, inputI, inputJ) * filters[f][d][k][l];
                                    }
                                }
                            }
                        }
                        sum += biases[f];
                        output[f][i][j] = activationFunction.activate(sum);
                    }
                }
            }
        }
    }

    @Override
    public Tensor backpropagate(Object input, Object gradientOutput) {
        if (!(input instanceof Tensor) || !(gradientOutput instanceof Tensor)) {
            throw new IllegalArgumentException("Expected input and gradientOutput to be Tensors.");
        }
        Tensor tensorInput = (Tensor) input;
        Tensor tensorGradientOutput = (Tensor) gradientOutput;

        Tensor gradientInput = new Tensor(inputDepth, inputHeight, inputWidth);

        POOL.invoke(new BackpropagationTask(tensorInput, tensorGradientOutput, gradientInput, 0, numFilters));

        return gradientInput;
    }

    private class BackpropagationTask extends RecursiveAction {
        private final Tensor input, gradientOutput, gradientInput;
        private final int startFilter, endFilter;

        BackpropagationTask(Tensor input, Tensor gradientOutput, Tensor gradientInput, int startFilter, int endFilter) {
            this.input = input;
            this.gradientOutput = gradientOutput;
            this.gradientInput = gradientInput;
            this.startFilter = startFilter;
            this.endFilter = endFilter;
        }

        @Override
        protected void compute() {
            if (endFilter - startFilter <= PARALLELISM_THRESHOLD) {
                backpropagateSequential();
            } else {
                int midFilter = (startFilter + endFilter) / 2;
                invokeAll(
                        new BackpropagationTask(input, gradientOutput, gradientInput, startFilter, midFilter),
                        new BackpropagationTask(input, gradientOutput, gradientInput, midFilter, endFilter)
                );
            }
        }

        private void backpropagateSequential() {
            for (int f = startFilter; f < endFilter; f++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        double gradientValue = gradientOutput.get(f, i, j) * activationFunction.derivative(gradientOutput.get(f, i, j));

                        gradientBiases[f] += gradientValue;

                        for (int d = 0; d < inputDepth; d++) {
                            for (int k = 0; k < filterSize; k++) {
                                for (int l = 0; l < filterSize; l++) {
                                    int inputI = i * strideY - paddingY + k;
                                    int inputJ = j * strideX - paddingX + l;
                                    if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                        double inputValue = input.get(d, inputI, inputJ);
                                        gradientFilters[f][d][k][l] += gradientValue * inputValue;
                                        synchronized (gradientInput) {
                                            gradientInput.set(d, inputI, inputJ, gradientInput.get(d, inputI, inputJ) + gradientValue * filters[f][d][k][l]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public void resetGradients() {
        for (int f = 0; f < numFilters; f++) {
            gradientBiases[f] = 0;
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    Arrays.fill(gradientFilters[f][d][i], 0);
                }
            }
        }
    }

    @Override
    public String toString() {
        return "ConvLayer: in:" + inputSize + "\tout:" + outputSize + "\tfilters:" + numFilters + "x" + filterSize + "x" + filterSize;
    }

    @Override
    public void dumpInfo() {
        System.out.println("Conv layer | filter average: " + getAverageFilterValue()+", range: "+getFilterMin()+", "+getFilterMax());
    }
}
