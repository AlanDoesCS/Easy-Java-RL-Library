package Structures;

import Tools.math;

import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.atomic.AtomicReference;

public class ConvLayer extends Layer {
    private static final int PARALLELISM_THRESHOLD = 128;   // threshold for parallelizing loops - increase value for weaker systems
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

    private float[][][][] filters; // [numFilters][depth][height][width]
    private float[] biases; // [numFilters]
    private int strideX, strideY;
    private int paddingX, paddingY;
    private int filterSize; // assumes square filters
    private int numFilters;
    private int inputWidth, inputHeight, inputDepth;
    private int outputWidth, outputHeight;

    private float[][][][] gradientFilters;
    private float[] gradientBiases;

    public ConvLayer(int inputWidth, int inputHeight, int inputDepth, int filterSize, int numFilters, int strideX, int strideY, int paddingX, int paddingY) {
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
        initializeParameters();
    }

    private void initializeParameters() {
        // Initialize filters with small random values
        for (int i = 0; i < filters.length; i++) {
            for (int j = 0; j < filters[i].length; j++) {
                for (int k = 0; k < filters[i][j].length; k++) {
                    for (int l = 0; l < filters[i][j][k].length; l++) {
                        filters[i][j][k][l] = math.randomFloat(-0.005f, 0.005f);
                    }
                }
            }
        }

        Arrays.fill(biases, 0);
    }

    public Matrix compute(Matrix input) {
        if (input.getWidth() != 1) {
            throw new IllegalArgumentException("Input must be a column matrix");
        }
        AtomicReference<float[]> outputRef = new AtomicReference<>(new float[outputSize]);
        POOL.invoke(new ComputeTask(input, outputRef, 0, numFilters));
        return new Matrix(outputRef.get(), outputHeight * outputWidth, numFilters);
    }

    private class ComputeTask extends RecursiveAction {
        private final Matrix input;
        private final AtomicReference<float[]> output;
        private final int startFilter, endFilter;

        ComputeTask(Matrix input, AtomicReference<float[]> output, int startFilter, int endFilter) {
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
            float[] outputArray = output.get();
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
                                        int inputIndex = (d * inputHeight * inputWidth) + (inputI * inputWidth) + inputJ;
                                        sum += input.get(0, inputIndex) * filters[f][d][k][l];
                                    }
                                }
                            }
                        }
                        sum += biases[f];
                        outputArray[f * outputWidth * outputHeight + i * outputWidth + j] = sum;
                    }
                }
            }
        }
    }

    @Override
    public Matrix backpropagate(Matrix input, Matrix gradientOutput) {
        Matrix gradientInput = new Matrix(inputSize, 1);

        // Reset gradients
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    Arrays.fill(gradientFilters[f][d][i], 0);
                }
            }
        }
        Arrays.fill(gradientBiases, 0);

        POOL.invoke(new BackpropagationTask(input, gradientOutput, gradientInput, 0, numFilters));

        return gradientInput;
    }

    private class BackpropagationTask extends RecursiveAction {
        private final Matrix input, gradientOutput, gradientInput;
        private final int startFilter, endFilter;

        BackpropagationTask(Matrix input, Matrix gradientOutput, Matrix gradientInput, int startFilter, int endFilter) {
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
                        int outputIndex = f * outputWidth * outputHeight + i * outputWidth + j;
                        float gradientValue = gradientOutput.get(0, outputIndex);

                        gradientBiases[f] += gradientValue;

                        for (int d = 0; d < inputDepth; d++) {
                            for (int k = 0; k < filterSize; k++) {
                                for (int l = 0; l < filterSize; l++) {
                                    int inputI = i * strideY - paddingY + k;
                                    int inputJ = j * strideX - paddingX + l;
                                    if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                        int inputIndex = d * inputWidth * inputHeight + inputI * inputWidth + inputJ;
                                        float inputValue = input.get(0, inputIndex);
                                        gradientFilters[f][d][k][l] += gradientValue * inputValue;
                                        gradientInput.add(inputIndex, 0, gradientValue * filters[f][d][k][l]);
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
    public void updateParameters(float learningRate) {
        for (int f = 0; f < numFilters; f++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        filters[f][d][i][j] -= learningRate * gradientFilters[f][d][i][j];
                    }
                }
            }
            biases[f] -= learningRate * gradientBiases[f];
        }
    }

    @Override
    public String toString() {
        return "ConvLayer: in:" + inputSize + "\tout:" + outputSize + "\tfilters:" + numFilters + "x" + filterSize + "x" + filterSize;
    }
}
