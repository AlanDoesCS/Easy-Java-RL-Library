package Structures;

import java.io.Serializable;
import java.util.List;

/**
 * Abstract base class for all layers in a neural network.
 * <p>
 * This class provides a common interface for different types of layers
 * and includes methods for computing outputs, copying parameters, and
 * performing backpropagation.
 * </p>
 */
public abstract class Layer implements Serializable {
    protected double alpha=0.001f; // LR for optimizers
    protected int inputSize;
    protected int outputSize;

    /**
     * Computes the output of the layer given an input.
     *
     * @param input the input to the layer.
     * @return the output of the layer.
     */
    public abstract Object compute(Object input);
    /**
     * Copies the current layer's parameters to the target layer.
     *
     * @param targetLayer the layer to which the parameters will be copied.
     * @param ignorePrimitives if true, primitive fields will not be copied.
     */
    public abstract void copyTo(Layer targetLayer, boolean ignorePrimitives);
    /**
     * Creates a copy of the current layer.
     *
     * @return a new instance of the Layer class that is a copy of the current layer.
     */
    public abstract Layer copy();
    public abstract String toString();

    public int getOutputSize() {
        return outputSize;
    }
    public int getInputSize() {
        return inputSize;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public static String toString(List<Layer> layers) {
        StringBuilder sb = new StringBuilder();
        for (Layer layer : layers) {
            sb.append(layer.toString()).append("\n");
        }
        return sb.toString();
    }

    /**
     * Performs the backpropagation algorithm on the layer.
     *
     * @param input the input to the layer.
     * @param gradientOutput the gradient of the loss with respect to the output of the layer.
     *                       - Can be either MatrixDouble or Tensor
     * @return the gradient of the loss with respect to the input of the layer.
     */
    public abstract Object backpropagate(Object input, Object gradientOutput);
    public abstract void updateParameters(double learningRate);

    public void dumpInfo() {}
}
