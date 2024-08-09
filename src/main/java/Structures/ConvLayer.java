package Structures;

import jdk.jshell.spi.ExecutionControl;

import java.util.Arrays;

public class ConvLayer extends Layer {
    private float[][][][] filters; // [numFilters][depth][height][width]
    private float[] biases; // [numFilters]
    private int strideX, strideY;
    private int paddingX, paddingY;
    private int filterSize; // assumes square filters
    private int numFilters;

    public ConvLayer(int inputWidth, int inputHeight, int inputDepth, int filterSize, int numFilters, int strideX, int strideY, int paddingX, int paddingY) {
        this.inputSize = inputWidth * inputHeight * inputDepth;
        this.filterSize = filterSize;
        this.numFilters = numFilters;
        this.strideX = strideX;
        this.strideY = strideY;
        this.paddingX = paddingX;
        this.paddingY = paddingY;

        int outputWidth = (inputWidth - filterSize + 2 * paddingX) / strideX + 1;
        int outputHeight = (inputHeight - filterSize + 2 * paddingY) / strideY + 1;
        this.outputSize = outputWidth * outputHeight * numFilters;

        // Initialize filters and biases
        filters = new float[numFilters][inputDepth][filterSize][filterSize];
        biases = new float[numFilters];
        initializeParameters();
    }

    private void initializeParameters() {
        // Initialize filters with small random values
        for (int i = 0; i < filters.length; i++) {
            for (int j = 0; j < filters[i].length; j++) {
                for (int k = 0; k < filters[i][j].length; k++) {
                    for (int l = 0; l < filters[i][j][k].length; l++) {
                        filters[i][j][k][l] = (float) (Math.random() - 0.5) * 0.01f;
                    }
                }
            }
        }

        Arrays.fill(biases, 0);
    }

    @Override
    public Matrix compute(Matrix input) {   // TODO: optimise for performance
        int inputWidth = (int) Math.cbrt(input.cols);
        int inputHeight = inputWidth;
        int inputDepth = input.cols / (inputWidth * inputHeight);

        int outputWidth = (inputWidth - filterSize + 2 * paddingX) / strideX + 1;
        int outputHeight = (inputHeight - filterSize + 2 * paddingY) / strideY + 1;

        Matrix output = new Matrix(outputWidth * outputHeight * numFilters, 1);

        for (int f = 0; f < numFilters; f++) {
            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    float sum = 0;
                    for (int d = 0; d < inputDepth; d++) {
                        for (int k = 0; k < filterSize; k++) {
                            for (int l = 0; l < filterSize; l++) {
                                int inputI = i * strideY - paddingY + k;
                                int inputJ = j * strideX - paddingX + l;
                                if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                    sum += input.get(d * inputWidth * inputHeight + inputI * inputWidth + inputJ, 0) * filters[f][d][k][l];
                                }
                            }
                        }
                    }
                    sum += biases[f];
                    output.set(f * outputWidth * outputHeight + i * outputWidth + j, 0, sum);
                }
            }
        }
        return output;
    }

    @Override
    public Matrix backpropagate(Matrix input, Matrix gradientOutput) {
        System.out.println("ConvLayer: backpropagate not implemented yet");
        return null;
    }

    @Override
    public void updateParameters(float learningRate) {
        System.out.println("ConvLayer: updateParameters not implemented yet");
    }

    @Override
    public String toString() {
        return "ConvLayer: in:" + inputSize + "\tout:" + outputSize + "\tfilters:" + numFilters + "x" + filterSize + "x" + filterSize;
    }
}
