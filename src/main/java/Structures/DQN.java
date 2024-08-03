package Structures;

import Training.ActivationFunction;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;

public class DQN extends NN {
    private final int inputSize, outputSize;

    public DQN(int inputSize, List<Layer> hiddenLayers, int outputSize, float learningRate, ActivationFunction outputActivation, float outputBias) {
        this.hiddenLayers  = hiddenLayers;
        this.outputLayer = new Layer(hiddenLayers.getLast().getOutputSize(), outputSize, outputActivation, outputBias);
        this.learningRate = learningRate;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public Matrix getOutput(Matrix input) {
        List<Matrix> layerInputs = forwardPass(input);
        return outputLayer.compute(layerInputs.getLast());
    }

    public void addLayer(int size, ActivationFunction phi, float bias) {
        hiddenLayers.add(new Layer(hiddenLayers.getLast().getOutputSize(), size, phi, bias));
    }

    @Override
    public void saveNN(String filename) {
        try (ObjectOutputStream outStream = new ObjectOutputStream(new FileOutputStream(filename))) {
            outStream.writeInt(hiddenLayers.size());
            for (Layer layer : hiddenLayers) {
                outStream.writeObject(layer.weights);
                outStream.writeObject(layer.biases);
            }

            outStream.writeObject(outputLayer.weights);
            outStream.writeObject(outputLayer.biases);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void loadNN(String filename) {
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filename))) {
            int hiddenLayerCount = inputStream.readInt();
            for (int i = 0; i < hiddenLayerCount; i++) {
                hiddenLayers.get(i).weights = (Matrix) inputStream.readObject();
                hiddenLayers.get(i).biases = (Matrix) inputStream.readObject();
            }

            outputLayer.weights = (Matrix) inputStream.readObject();
            outputLayer.biases = (Matrix) inputStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    /*
    -----------------------------------------------------------------------------

    ACCESSORS AND MUTATORS

    -----------------------------------------------------------------------------
    */
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
