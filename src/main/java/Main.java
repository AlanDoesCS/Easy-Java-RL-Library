import Structures.*;
import Tools.DQN_Visualiser;
import Tools.Environment_Visualiser;
import Tools.Pathfinding.Pathfinder;
import Training.*;
import com.sun.jdi.InvalidTypeException;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Environment.setDimensions(500, 500);

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class, MazeGridEnvironment.class, RandomGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        ActivationFunction sig = new Sigmoid();
        ActivationFunction relu = new ReLU();

        List<Layer> layers = List.of(
                new Layer(trainer.getStateSpace(), 512, sig, 0),
                new Layer(512, 512, sig, 0),
                new Layer(512, 512, sig, 0)
        );

        DQNAgent dqnAgent = new DQNAgent(4, layers, 0.1f, 0.99f, 0.001f, relu, 0);

        trainer.trainAgent(dqnAgent, 6000, 100);
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, new Color(0, 128, 128));
    }
}