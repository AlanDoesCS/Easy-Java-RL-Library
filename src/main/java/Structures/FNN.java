package Structures;

import java.util.List;

public class FNN {

    private float learningRate;
    private Matrix inputNodes, outputNodes;
    List<Layer> hiddenLayers;
    Layer outputLayer;

    public FNN(int inputNodes, int outputNodes, float learningRate) {
        this.inputNodes = new Matrix(inputNodes, 1);
        this.outputNodes = new Matrix(outputNodes, 1);
        this.learningRate = learningRate;
    }

    public Matrix getOutput(Matrix input) {
        for (Layer layer : hiddenLayers) {
            input = layer.compute(input);
        }
        return outputLayer.compute(input); // is output after all layers
    }
}
