package Tools.Presets;

import Structures.*;
import Training.Environment;
import Training.LeakyReLU;
import Training.Linear;
import Training.ReLU;

import java.util.ArrayList;
import java.util.List;

public class DQNAgents {
    private static DDQNAgent GRID_DQN_FROM_LAYERS(List<Layer> layers) {
        return new DDQNAgent(
                Environment.getActionSpace(),   // action space
                layers,                         // layers
                1f,                             // initial epsilon
                0.9999995f,                     // epsilon decay
                0.01f,                          // epsilon min
                0.9999f,                        // gamma
                0.001f,                         // learning rate
                0.9999995f,                     // learning rate decay
                0.00001f,                       // learning rate minimum
                0.0001f                         // tau
        );
    }

    public static DDQNAgent LARGE_GRID_DQN_AGENT() {
        List<Layer> layers = new ArrayList<>();
        ReLU activation = new ReLU();

        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 32, 3, 64, 1, 1, 1, 1));
        layers.add(new FlattenLayer(64, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(64 * Environment.getGridWidth() * Environment.getGridHeight(), 256, new ReLU(), 0));
        layers.add(new MLPLayer(256, Environment.getActionSpace(), new Linear(), 0));

        return GRID_DQN_FROM_LAYERS(layers);
    }

    public static DDQNAgent MEDIUM_GRID_DQN_AGENT() {
        List<Layer> layers = new ArrayList<>();
        ReLU activation = new ReLU();

        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new FlattenLayer(32, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(32 * Environment.getGridWidth() * Environment.getGridHeight(), 128, new ReLU(), 0));
        layers.add(new MLPLayer(128, Environment.getActionSpace(), new Linear(), 0));

        return GRID_DQN_FROM_LAYERS(layers);
    }

    public static DDQNAgent MLP_ONLY_GRID_DQN_AGENT() {
        List<Layer> layers = new ArrayList<>();
        LeakyReLU activation = new LeakyReLU(0.1f);

        layers.add(new FlattenLayer(3, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(3 * Environment.getGridWidth() * Environment.getGridHeight(), 16, activation, 0));
        layers.add(new MLPLayer(16, Environment.getActionSpace(), new Linear(), 0));

        return GRID_DQN_FROM_LAYERS(layers);
    }
}
