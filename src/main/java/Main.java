import Structures.*;
import Tools.DQN_Visualiser;
import Tools.Environment_Visualiser;
import Tools.Pathfinding.Pathfinder;
import Training.*;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {

        final int width=720, height=720;

        PerlinGridEnvironment environment = new PerlinGridEnvironment(width, height, 8, 0.9f, 0.01f);

        System.out.println(environment.getState());

        testPathfinding(environment.getAgentPosition(), environment.getGoalPosition(), environment);

        ActivationFunction sig = new Sigmoid();
        ActivationFunction relu = new ReLU();

        List<Layer> layers = List.of(
                new Layer(environment.getStateSpace(), 512, sig, 0),
                new Layer(512, 512, sig, 0),
                new Layer(512, 512, sig, 0)
        );

        DQNAgent dqnAgent = new DQNAgent(4, layers, 0.1f, 0.99f, 0.001f, relu, 0);
    }

    public static void testPathfinding(Vector2D start, Vector2D end, PerlinGridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.YELLOW);
    }
}