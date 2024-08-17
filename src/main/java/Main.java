import Structures.*;
import Tools.Environment_Visualiser;
import Tools.GraphPlotter;
import Tools.Pathfinding.Pathfinder;
import Tools.Perlin1D;
import Tools.Presets.DQNAgents;
import Training.*;

import com.sun.jdi.InvalidTypeException;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Environment.setDimensions(10, 10);
        Environment.setActionSpace(5);

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class, RandomGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        // DQNAgent dqnAgent = DQNAgents.MEDIUM_GRID_DQN_AGENT();

        List<Layer> layers = new ArrayList<>();
        ReLU activation = new ReLU();

        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(activation, Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new FlattenLayer(32, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(32 * Environment.getGridWidth() * Environment.getGridHeight(), 128, new ReLU(), 0));
        layers.add(new MLPLayer(128, Environment.getActionSpace(), new Linear(), 0));

        DQNAgent dqnAgent = new DQNAgent(
                Environment.getActionSpace(),
                layers,
                1f,
                0.999995f,
                0.01f,
                0.9999f,
                0.0001f
        );

        trainer.trainAgent(dqnAgent, 600000, 500, 1, "plot", "ease", "axis_ticks", "show_path");
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.RED);
    }
}