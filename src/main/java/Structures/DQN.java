package Structures;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

public class DQN extends NN {
    private final int inputSize, outputSize;

    public DQN(int inputSize, List<Layer> layers, float learningRate) {
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

    public int numLayers() {
        return layers.size();
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
}
