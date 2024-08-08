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
        int octaves = 10;
        float persistence = 0.8f;

        Environment.setDimensions(20, 20);
        Environment.setStateSpace(Environment.getGridSquares()+4);
        Environment.setActionSpace(5);

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class, MazeGridEnvironment.class, RandomGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        ActivationFunction sig = new Sigmoid();
        ActivationFunction relu = new ReLU();

        List<Layer> layers = Layer.createHiddenLayers(
                List.of(100, 200, 300),
                List.of(sig, sig, sig),
                List.of(0, 0, 0)
        );

        DQNAgent dqnAgent = new DQNAgent(Environment.getActionSpace(), layers, 0.1f, 0.99f, 0.001f, relu, 0);

        trainer.trainAgent(dqnAgent, 6000, 1, "plot", "ease", "axis_ticks", "verbose");
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.RED);//new Color(0, 128, 128));
    }
}