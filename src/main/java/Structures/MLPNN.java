package Structures;

import java.util.List;

public class MLPNN extends NN {
    private final int inputSize, outputSize;

    public MLPNN(int inputSize, List<Layer> layers, float learningRate) {
        this.layers = layers;
        this.learningRate = learningRate;
        this.inputSize = inputSize;
        this.outputSize = layers.getLast().getOutputSize();
    }

    @Override
    public Object getOutput(Object input) {
        return forwardPass(input).getLast();
    }

    @Override
    public void saveNN(String filename) {

    }

    @Override
    public void loadNN(String filename) {

    }
}
