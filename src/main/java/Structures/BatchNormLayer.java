package Structures;

public class BatchNormLayer extends Layer {
    private int depth, height, width;
    private float epsilon = 1e-5f;
    private float momentum = 0.99f;

    private float[] gamma;
    private float[] beta;
    private float[] runningMean;
    private float[] runningVar;

    public BatchNormLayer(int depth, int height, int width) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.inputSize = depth * height * width;
        this.outputSize = inputSize;

        gamma = new float[depth];
        beta = new float[depth];
        runningMean = new float[depth];
        runningVar = new float[depth];

        for (int i = 0; i < depth; i++) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
        }
    }

    @Override
    public Object compute(Object input) {
        if (!(input instanceof Tensor inputTensor)) {
            throw new IllegalArgumentException("Expected input to be a Tensor.");
        }
        Tensor outputTensor = new Tensor(depth, height, width);

        for (int d = 0; d < depth; d++) {
            float mean = 0, variance = 0;
            int count = height * width;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    mean += inputTensor.get(d, h, w);
                }
            }
            mean /= count;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float diff = inputTensor.get(d, h, w) - mean;
                    variance += diff * diff;
                }
            }
            variance /= count;

            runningMean[d] = momentum * runningMean[d] + (1 - momentum) * mean;
            runningVar[d] = momentum * runningVar[d] + (1 - momentum) * variance;

            // Normalize and scale
            float stdDev = (float) Math.sqrt(variance + epsilon);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    float normalized = (inputTensor.get(d, h, w) - mean) / stdDev;
                    outputTensor.set(d, h, w, gamma[d] * normalized + beta[d]);
                }
            }
        }

        return outputTensor;
    }

    @Override
    public Object backpropagate(Object input, Object gradientOutput) {
        // TODO: Implement backpropagation
        return gradientOutput;
    }

    @Override
    public void updateParameters(float learningRate) {
    }

    @Override
    public void copyTo(Layer targetLayer, boolean ignorePrimitives) {
        if (!(targetLayer instanceof BatchNormLayer target)) {
            throw new IllegalArgumentException("Target layer must be a BatchNormLayer");
        }

        System.arraycopy(this.gamma, 0, target.gamma, 0, this.gamma.length);
        System.arraycopy(this.beta, 0, target.beta, 0, this.beta.length);
        System.arraycopy(this.runningMean, 0, target.runningMean, 0, this.runningMean.length);
        System.arraycopy(this.runningVar, 0, target.runningVar, 0, this.runningVar.length);

        if (!ignorePrimitives) {
            target.depth = this.depth;
            target.height = this.height;
            target.width = this.width;
            target.inputSize = this.inputSize;
            target.outputSize = this.outputSize;
            target.epsilon = this.epsilon;
            target.momentum = this.momentum;
        }
    }

    @Override
    public Layer copy() {
        BatchNormLayer copy = new BatchNormLayer(depth, height, width);
        copyTo(copy, false);
        return copy;
    }

    @Override
    public String toString() {
        return "BatchNormLayer: in:" + inputSize + " out:" + outputSize;
    }
}
