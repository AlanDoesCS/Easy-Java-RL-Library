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
        int octaves = 8;
        float persistence = 0.4f;

        Environment.setDimensions(10, 10);
        Environment.setStateSpace(Environment.getGridSquares()+4);
        Environment.setActionSpace(4);

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

        // trainer.trainAgent(dqnAgent, 6000, 1, "plot", "ease");

        GraphPlotter plotter = new GraphPlotter("Test Plot (y = log x)", GraphPlotter.Types.LINE, "X-Axis", "Y-Axis", "axis_ticks");
        plotter.setVisible(true);

        Perlin1D p = new Perlin1D(octaves, persistence);

        GraphPlotter plotter1 = new GraphPlotter("Perlin Noise 1D", GraphPlotter.Types.LINE, "X-Axis", "Y-Axis");
        plotter1.setVisible(true);

        for (float i = 0; i < 10; i+=0.01f) {
            plotter.addPoint(new Vector2D(i, (float) Math.log(i+0.1f)));
            plotter1.addPoint(new Vector2D(i, p.noise(i)));
        }
        plotter.plot();
        plotter1.plot();

        GraphPlotter plotter2 = new GraphPlotter("Test Plot 2 (y = tan(sin(cos(x^2)))^5)", GraphPlotter.Types.LINE, "X-Axis", "Y-Axis", "axis_ticks");
        plotter2.setVisible(true);
        plotter2.plot(Main::func, 0f, 10f, 0.01f);
    }

    public static Float func(Float x) {
        return (float) Math.pow(Math.tan(Math.sin(Math.cos(x * x))), 5);
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, new Color(0, 128, 128));
    }
}