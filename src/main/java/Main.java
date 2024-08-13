import Structures.*;
import Tools.Environment_Visualiser;
import Tools.GraphPlotter;
import Tools.Pathfinding.Pathfinder;
import Tools.Perlin1D;
import Training.*;

import com.sun.jdi.InvalidTypeException;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Environment.setDimensions(20, 20);
        Environment.setActionSpace(5);

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class, MazeGridEnvironment.class, RandomGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        List<Layer> layers = new ArrayList<>();
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new ConvLayer(Environment.getGridWidth(), Environment.getGridHeight(), 32, 3, 64, 1, 1, 1, 1));
        ConvLayer lastConvLayer = (ConvLayer) layers.getLast();
        layers.add(new FlattenLayer(lastConvLayer.getOutputDepth(), lastConvLayer.getOutputHeight(), lastConvLayer.getOutputWidth()));
        layers.add(new MLPLayer(64 * Environment.getGridWidth() * Environment.getGridHeight(), 256, new ReLU(), 0));
        layers.add(new MLPLayer(256, Environment.getActionSpace(), new Linear(), 0));

        DQNAgent dqnAgent = new DQNAgent(
                Environment.getActionSpace(),
                layers,
                1f,
                0.995f,
                0.01f,
                0.99f,
                0.001f
        );

        trainer.trainAgent(dqnAgent, 6000, 1000, 10, "plot", "ease", "axis_ticks", "verbose");
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.RED);
    }
}