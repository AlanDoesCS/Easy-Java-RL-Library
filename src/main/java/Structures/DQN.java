package Structures;

import Training.ActivationFunction;

import java.util.List;

public class DQN {

    private float learningRate, bias;
    private final int inputSize, outputSize;
    private final List<Layer> hiddenLayers;
    private final Layer outputLayer;

    public DQN(int inputSize, List<Layer> hiddenLayers, int outputSize, float learningRate, ActivationFunction phi, float bias) {
        this.hiddenLayers  = hiddenLayers;
        this.outputLayer = new Layer(hiddenLayers.getLast().getOutputSize(), outputSize, phi, bias);
        this.learningRate = learningRate;
        this.bias = bias;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public Matrix getOutput(Matrix input) {
        for (Layer layer : hiddenLayers) {
            input = layer.compute(input);
        }
        return outputLayer.compute(input); // is output after all layers
    }

    public int layers() {
        return hiddenLayers.size()+2; //input + hidden + output
    }

    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public Layer getHiddenLayer(int i) {
        return hiddenLayers.get(i);
    }

    public Layer getOutputLayer() {
        return outputLayer;
    }
}
