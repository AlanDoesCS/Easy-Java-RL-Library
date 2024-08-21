package Structures;

import Training.Optimizers.Optimizer;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

public class DQN extends NN {
    private final int inputSize, outputSize;
    private Optimizer optimizer;

    public DQN(int inputSize, List<Layer> layers, double learningRate) {
        this.layers = layers;
        this.learningRate = learningRate;
        this.inputSize = inputSize;
        this.outputSize = layers.getLast().getOutputSize();
    }

    @Override
    public Object getOutput(Object input) {
        List<Object> layerOutputs = forwardPass(input);
        return layerOutputs.getLast();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    @Override
    public void saveNN(String filename) {
        try (ObjectOutputStream outStream = new ObjectOutputStream(new FileOutputStream(filename))) {
            outStream.writeInt(layers.size());

            for (Layer layer : layers) {
                outStream.writeObject(layer);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadNN(String filename) {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filename))) {
            int layerCount = inputStream.readInt();

            for (int i = 0; i < layerCount; i++) {
                layers.set(i, (Layer) inputStream.readObject());
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void backpropagate(Object input, MatrixDouble target, List<Object> layerOutputs) {
        MatrixDouble output = (MatrixDouble) layerOutputs.getLast();
        Object gradientOutput = MatrixDouble.subtract(target, output);

        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Object layerInput = layerOutputs.get(i);

            gradientOutput = currentLayer.backpropagate(layerInput, gradientOutput);

            optimizer.optimize(currentLayer);
        }
    }

    public int numLayers() {
        return layers.size();
    }

    public static void copyNetworkWeightsAndBiases(DQN sourceNetwork, DQN targetNetwork) {
        if (sourceNetwork.numLayers() != targetNetwork.numLayers()) {
            throw new IllegalArgumentException(String.format("Source and target networks must have the same number of layers. (%d != %d)", sourceNetwork.numLayers(), targetNetwork.numLayers()));
        }

        for (int i = 0; i < sourceNetwork.numLayers(); i++) {
            sourceNetwork.getLayer(i).copyTo(targetNetwork.getLayer(i), false);
        }
    }



    public int getInputSize() {
        return inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public Layer getLayer(int i) {
        return layers.get(i);
    }

    public Layer getOutputLayer() {
        return layers.getLast();
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }
}
