import Structures.*;
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
        Environment.setDimensions(10, 10);
        Environment.setActionSpace(5);

        DDQNAgentTrainer trainer;
        try {
            trainer = new DDQNAgentTrainer(Set.of(PerlinGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        List<Layer> layers = new ArrayList<>();
        LeakyReLU leakyRelu = new LeakyReLU(0.1f);
        float lambda = 0.001f;

        // StateSpace is 104, ActionSpace is 5
        layers.add(new MLPLayer(Environment.getStateSpace(), 64, leakyRelu, 0, lambda));
        layers.add(new BatchNormLayer(64, 1, 1));
        layers.add(new MLPLayer(64, 128, leakyRelu, 0, lambda));
        layers.add(new BatchNormLayer(128, 1, 1));
        layers.add(new MLPLayer(128, Environment.getActionSpace(), new Linear(), 0, lambda));

        DDQNAgent ddqnAgent = new DDQNAgent(
                Environment.getActionSpace(),   // action space
                layers,                         // layers
                1f,                             // initial epsilon
                0.999995f,                      // epsilon decay
                0.01f,                          // epsilon min
                0.9999f,                        // gamma
                0.0001f,                        // learning rate
                0.999995f,                      // learning rate decay
                0.00001f,                       // learning rate minimum
                0.001f                          // tau
        );

        ddqnAgent.dumpDQNInfo();
        trainer.trainAgent(ddqnAgent, 600000, 500, 1, "plot", "ease", "axis_ticks", "show_path");
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.RED);
    }
}