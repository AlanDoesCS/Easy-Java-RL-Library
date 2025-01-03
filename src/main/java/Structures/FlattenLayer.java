package Structures;

/**
 * Represents a Flatten layer in a neural network.
 * <p>
 * This class extends the abstract Layer class and provides functionality
 * for flattening a multi-dimensional input tensor into a 2D matrix.
 * </p>
 */
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
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        if (!(targetLayer instanceof FlattenLayer target)) {
            throw new IllegalArgumentException(String.format("Target layer must be a FlattenLayer (got: %s)", targetLayer.getClass().getSimpleName()));
        }

        if (ignorePrimitives) return;

        target.inputSize = this.inputSize;
        target.outputSize = this.outputSize;

        target.inputDepth = this.inputDepth;
        target.inputHeight = this.inputHeight;
        target.inputWidth = this.inputWidth;
    }

    @Override
    public FlattenLayer copy() {
        return new FlattenLayer(inputDepth, inputHeight, inputWidth);
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

        MatrixDouble output = new MatrixDouble(outputSize, 1);
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
        if (!(gradientOutput instanceof MatrixDouble matrixGradientOutput)) {
            throw new IllegalArgumentException("Expected gradientOutput to be a MatrixDouble.");
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
    public void resetGradients() {
        // unused
    }

    @Override
    public String toString() {
        return "FlattenLayer: in:" + inputSize + "\tout:" + outputSize;
    }
}
