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

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        // DDQNAgent DDQNAgent = DQNAgents.MEDIUM_GRID_DQN_AGENT();

        List<Layer> layers = new ArrayList<>();
        LeakyReLU leakyRelu = new LeakyReLU(0.1f);

        layers.add(new ConvLayer(leakyRelu, Environment.getGridWidth(), Environment.getGridHeight(), 3, 3, 16, 1, 1, 1, 1));
        layers.add(new BatchNormLayer(16, Environment.getGridWidth(), Environment.getGridHeight()));
        layers.add(new ConvLayer(leakyRelu, Environment.getGridWidth(), Environment.getGridHeight(), 16, 3, 32, 1, 1, 1, 1));
        layers.add(new BatchNormLayer(32, Environment.getGridWidth(), Environment.getGridHeight()));
        layers.add(new FlattenLayer(32, Environment.getGridHeight(), Environment.getGridWidth()));
        layers.add(new MLPLayer(32 * Environment.getGridWidth() * Environment.getGridHeight(), 256, leakyRelu, 0));
        layers.add(new MLPLayer(256, Environment.getActionSpace(), new Linear(), 0));

        DDQNAgent DDQNAgent = new DDQNAgent(
                Environment.getActionSpace(),   // action space
                layers,                         // layers
                1f,                             // initial epsilon
                0.999995f,                       // epsilon decay
                0.01f,                          // epsilon min
                0.9999f,                        // gamma
                0.0001f,                        // learning rate
                0.999995f,                      // learning rate decay
                0.00001f,                       // learning rate minimum
                0.001f                          // tau
        );

        DDQNAgent.dumpDQNInfo();

        trainer.trainAgent(DDQNAgent, 600000, 500, 1, "plot", "ease", "axis_ticks", "show_path");
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.RED);
    }
}