package Structures;

import Tools.math;
import Training.ActivationFunction;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ConvLayer extends Layer {
    private static final int PARALLELISM_THRESHOLD = 32;   // threshold for parallelizing loops - increase value for weaker systems
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

    float[][][][] filters; // [numFilters][depth][height][width]
    float[] biases; // [numFilters]
    private int strideX, strideY;
    private int paddingX, paddingY;
    public int filterSize; // assumes square filters
    private int numFilters;
    int inputWidth;
    int inputHeight;
    int inputDepth;
    private int outputWidth, outputHeight;

    private ActivationFunction activationFunction;

    private static final float clipValue = 5.0f;

    private float[][][][] gradientFilters;
    private float[] gradientBiases;

    @Override
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        if (!(targetLayer instanceof ConvLayer target)) {
            throw new IllegalArgumentException(String.format("Target layer must be a ConvLayer (got: %s)", targetLayer.getClass().getSimpleName()));
        }
        for (int f = 0; f < this.filters.length; f++) {
            for (int d = 0; d < this.filters[f].length; d++) {
                for (int h = 0; h < this.filters[f][d].length; h++) {
                    System.arraycopy(this.filters[f][d][h], 0, target.filters[f][d][h], 0, this.filters[f][d][h].length);
                }
            }
        }
        for (int f = 0; f < this.gradientFilters.length; f++) {
            for (int d = 0; d < this.gradientFilters[f].length; d++) {
                for (int h = 0; h < this.gradientFilters[f][d].length; h++) {
                    System.arraycopy(this.gradientFilters[f][d][h], 0, target.gradientFilters[f][d][h], 0, this.gradientFilters[f][d][h].length);
                }
            }
        }

        System.arraycopy(this.biases, 0, target.biases, 0, this.biases.length);
        System.arraycopy(this.gradientBiases, 0, target.gradientBiases, 0, this.gradientBiases.length);

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
        target.activationFunction = this.activationFunction;
    }

    @Override
    public Layer copy() {
        ConvLayer copyLayer = new ConvLayer(activationFunction, inputWidth, inputHeight, inputDepth, filterSize, numFilters, strideX, strideY, paddingX, paddingY, "noInit");
        copyTo(copyLayer, true);
        return copyLayer;
    }

    public ConvLayer(ActivationFunction activationFunction, int inputWidth, int inputHeight, int inputDepth, int filterSize, int numFilters, int strideX, int strideY, int paddingX, int paddingY, String... args) {
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
        filters = new float[numFilters][inputDepth][filterSize][filterSize];
        biases = new float[numFilters];
        gradientFilters = new float[numFilters][inputDepth][filterSize][filterSize];
        gradientBiases = new float[numFilters];

        if (args.length > 0) {
            List<String> argList = List.of(args);

            if (argList.contains("noInit")) {
                return;
            }
        }

        initializeParameters();
    }

    private void initializeParameters() {
        // He initialization (for ReLU)
        float stdDev = (float) Math.sqrt(2.0 / (inputDepth * filterSize * filterSize));
        for (int i = 0; i < filters.length; i++) {
            for (int j = 0; j < filters[i].length; j++) {
                for (int k = 0; k < filters[i][j].length; k++) {
                    for (int l = 0; l < filters[i][j][k].length; l++) {
                        filters[i][j][k][l] = math.randomFloat(-stdDev, stdDev);
                    }
                }
            }
        }

        Arrays.fill(biases, 0);
    }

    @Override
    public Object compute(Object input) {
        if (!(input instanceof Tensor tensorInput)) {
            throw new IllegalArgumentException("Expected input to be a Tensor, instead got: " + input.getClass().getSimpleName());
        }

        if (tensorInput.getDepth() != inputDepth || tensorInput.getHeight() != inputHeight || tensorInput.getWidth() != inputWidth) {
            throw new IllegalArgumentException("Input dimensions do not match expected dimensions: Expected: (" + inputDepth + ", " + inputHeight + ", " + inputWidth + "), Got: (" + tensorInput.getDepth() + ", " + tensorInput.getHeight() + ", " + tensorInput.getWidth() + ")");
        }

        float[][][] outputData = new float[numFilters][outputHeight][outputWidth];
        POOL.invoke(new ComputeTask(tensorInput, outputData, 0, numFilters));
        return new Tensor(outputData);
    }

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

    public float getFilterMin() {
        float min = Float.MAX_VALUE;
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
    public float getFilterMax() {
        float max = Float.MIN_VALUE;
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
    public float getAverageFilterValue() {
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
        return (float) (sum / (numFilters * inputDepth * filterSize * filterSize));
    }

    private class ComputeTask extends RecursiveAction {
        private final Tensor input;
        private final float[][][] output;
        private final int startFilter, endFilter;

        ComputeTask(Tensor input, float[][][] output, int startFilter, int endFilter) {
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
                        float sum = 0;
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
        if (!(input instanceof Tensor tensorInput)) {
            throw new IllegalArgumentException("Expected input to be a Tensor.");
        }
        if (!(gradientOutput instanceof Tensor tensorGradientOutput)) {
            throw new IllegalArgumentException("Expected gradientOutput to be a Tensor.");
        }

        Tensor gradientInput = new Tensor(inputDepth, inputHeight, inputWidth);

        // Reset gradients
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    Arrays.fill(gradientFilters[f][d][i], 0);
                }
            }
        }
        Arrays.fill(gradientBiases, 0);

        POOL.invoke(new BackpropagationTask(tensorInput, tensorGradientOutput, gradientInput, 0, numFilters));

        return gradientInput;
    }

    private class BackpropagationTask extends RecursiveAction {
        private final Tensor input, gradientOutput, gradientInput;
        private final int startFilter, endFilter;
        private final float[][][][] localGradientFilters;
        private final float[] localGradientBiases;

        BackpropagationTask(Tensor input, Tensor gradientOutput, Tensor gradientInput, int startFilter, int endFilter) {
            this.input = input;
            this.gradientOutput = gradientOutput;
            this.gradientInput = gradientInput;
            this.startFilter = startFilter;
            this.endFilter = endFilter;
            this.localGradientFilters = new float[endFilter - startFilter][inputDepth][filterSize][filterSize];
            this.localGradientBiases = new float[endFilter - startFilter];
        }

        @Override
        protected void compute() {
            if (endFilter - startFilter <= PARALLELISM_THRESHOLD) {
                backpropagateSequential();
            } else {
                int midFilter = (startFilter + endFilter) / 2;
                BackpropagationTask leftTask = new BackpropagationTask(input, gradientOutput, gradientInput, startFilter, midFilter);
                BackpropagationTask rightTask = new BackpropagationTask(input, gradientOutput, gradientInput, startFilter, midFilter);
                invokeAll(
                        leftTask,
                        rightTask
                );

                synchronized (ConvLayer.this) {
                    addLocalGradients(leftTask.localGradientFilters, leftTask.localGradientBiases, startFilter);
                    addLocalGradients(rightTask.localGradientFilters, rightTask.localGradientBiases, midFilter);
                }
            }
        }

        private void backpropagateSequential() {
            for (int f = startFilter; f < endFilter; f++) {
                for (int i = 0; i < outputHeight; i++) {
                    for (int j = 0; j < outputWidth; j++) {
                        float gradientValue = gradientOutput.get(f, i, j) * activationFunction.derivative(gradientOutput.get(f, i, j));

                        localGradientBiases[f - startFilter] += gradientValue;

                        for (int d = 0; d < inputDepth; d++) {
                            for (int k = 0; k < filterSize; k++) {
                                for (int l = 0; l < filterSize; l++) {
                                    int inputI = i * strideY - paddingY + k;
                                    int inputJ = j * strideX - paddingX + l;
                                    if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                        float inputValue = input.get(d, inputI, inputJ);
                                        localGradientFilters[f - startFilter][d][k][l] += gradientValue * inputValue;
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
            synchronized (ConvLayer.this) {
                addLocalGradients(localGradientFilters, localGradientBiases, startFilter);
            }
        }

        private void addLocalGradients(float[][][][] localFilters, float[] localBiases, int offset) {
            for (int f = 0; f < localFilters.length; f++) {
                for (int d = 0; d < inputDepth; d++) {
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            gradientFilters[f + offset][d][i][j] += localFilters[f][d][i][j];
                        }
                    }
                }
                gradientBiases[f + offset] += localBiases[f];
            }
        }
    }

    @Override
    public void updateParameters(float learningRate) {
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        gradientFilters[f][d][i][j] = math.clamp(gradientFilters[f][d][i][j], -clipValue, clipValue);
                        filters[f][d][i][j] -= learningRate * gradientFilters[f][d][i][j];
                    }
                }
            }
            gradientBiases[f] = math.clamp(gradientBiases[f], -clipValue, clipValue); // Clip the bias gradient
            biases[f] -= learningRate * gradientBiases[f];
        }
    }

    @Override
    public String toString() {
        return "ConvLayer: in:" + inputSize + "\tout:" + outputSize + "\tfilters:" + numFilters + "x" + filterSize + "x" + filterSize;
    }
}
