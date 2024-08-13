package Structures;

public class FlattenLayer extends Layer {
    private int inputDepth, inputHeight, inputWidth;

    public FlattenLayer(int inputDepth, int inputHeight, int inputWidth) {
        this.inputDepth = inputDepth;
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputSize = inputDepth * inputHeight * inputWidth;
        this.outputSize = inputSize;
    }

    @Override
    public Object compute(Object input) {
        if (!(input instanceof Tensor tensorInput)) {
            throw new IllegalArgumentException("Expected input to be a Tensor.");
        }

        if (tensorInput.getDepth() != inputDepth ||
                tensorInput.getHeight() != inputHeight ||
                tensorInput.getWidth() != inputWidth) {
            throw new IllegalArgumentException("Input dimensions do not match expected dimensions. "+
                    String.format(
                            "Expected: (%d, %d, %d), Received: (%d, %d, %d)",
                            inputDepth, inputHeight, inputWidth, tensorInput.getDepth(), tensorInput.getHeight(), tensorInput.getWidth()
                    )
            );
        }

        Matrix output = new Matrix(outputSize, 1);
        int index = 0;
        for (int d = 0; d < inputDepth; d++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    output.set(0, index++, tensorInput.get(d, h, w));
                }
            }
        }
        return output;
    }

    @Override
    public Object backpropagate(Object input, Object gradientOutput) {
        if (!(input instanceof Tensor tensorInput)) {
            throw new IllegalArgumentException("Expected input to be a Tensor.");
        }
        if (!(gradientOutput instanceof Matrix matrixGradientOutput)) {
            throw new IllegalArgumentException("Expected gradientOutput to be a Matrix.");
        }

        Tensor gradientInput = new Tensor(inputDepth, inputHeight, inputWidth);
        int index = 0;
        for (int d = 0; d < inputDepth; d++) {
            for (int h = 0; h < inputHeight; h++) {
                for (int w = 0; w < inputWidth; w++) {
                    gradientInput.set(d, h, w, matrixGradientOutput.get(0, index++));
                }
            }
        }
        return gradientInput;
    }

    @Override
    public void updateParameters(float learningRate) {
        // No parameters to update
    }

    @Override
    public String toString() {
        return "FlattenLayer: in:" + inputSize + "\tout:" + outputSize;
    }
}
