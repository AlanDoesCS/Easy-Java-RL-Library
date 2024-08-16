package Tools.Presets;

import Structures.*;
import Training.Environment;
import Training.Linear;
import Training.ReLU;

import java.util.ArrayList;
import java.util.List;

public class DQNAgents {
    private static DQNAgent GRID_DQN_FROM_LAYERS(List<Layer> layers) {
        return new DQNAgent(
                Environment.getActionSpace(),
                layers,
                1f,
                0.9999f,
                0.01f,
                0.99f,
                0.001f
        );
    }

    public static DQNAgent LARGE_GRID_DQN_AGENT() {
        List<Layer> layers = new ArrayList<>();

        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 32, 3, 64, 1, 1, 1, 1));
        layers.add(new FlattenLayer(64, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(64 * Environment.getGridWidth() * Environment.getGridHeight(), 256, new ReLU(), 0));
        layers.add(new MLPLayer(256, Environment.getActionSpace(), new Linear(), 0));

        return GRID_DQN_FROM_LAYERS(layers);
    }

    public static DQNAgent MEDIUM_GRID_DQN_AGENT() {
        List<Layer> layers = new ArrayList<>();

        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new FlattenLayer(32, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(32 * Environment.getGridWidth() * Environment.getGridHeight(), 128, new ReLU(), 0));
        layers.add(new MLPLayer(128, Environment.getActionSpace(), new Linear(), 0));

        return GRID_DQN_FROM_LAYERS(layers);
    }
}
