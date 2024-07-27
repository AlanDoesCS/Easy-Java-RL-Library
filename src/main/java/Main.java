import Structures.DQN;
import Structures.Layer;
import Structures.Matrix;
import Structures.Vector2D;
import Tools.DQN_Visualiser;
import Tools.Environment_Visualiser;
import Tools.Pathfinding.Pathfinder;
import Training.*;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        Random rand = new Random();

        final int width=720, height=720;

        PerlinGridEnvironment environment = new PerlinGridEnvironment(width, height, 8, 0.9f, 0.01f);
        int numSquares = environment.getNumSquares();

        int startX=rand.nextInt(width+1), startY=rand.nextInt(height+1);
        int endX=rand.nextInt(width+1), endY=rand.nextInt(height+1);

        Matrix input = new Matrix(new float[numSquares+4][1]); // Takes all square weights and start/end positions as parameters
        for (int i = 0; i < numSquares; i++) {
            input.set(0, i, environment.get(i));
        }
        // Start Position:
        input.set(0, numSquares, startX);
        input.set(0, numSquares+1, startY);

        // End Position:
        input.set(0, numSquares+2, endX);
        input.set(0, numSquares+3, endY);

        System.out.println(input);

        //testPathfinding(environment.getRandomCoordinateInBounds(), environment.getRandomCoordinateInBounds(), environment);

        ActivationFunction sig = new Sigmoid();
        List<Layer> layers = List.of(
                new Layer(input.getHeight(), 512, sig, 0),
                new Layer(512, 512, sig, 0),
                new Layer(512, 512, sig, 0)
        );

        DQN net = new DQN(numSquares, layers, 4, 1, sig, 0);

        for (int i = 0; i < 5; i++) {
            System.out.println("Run " + (i+1) + ":");
            Matrix output = net.getOutput(input);
            System.out.println(output);
            System.out.println();
        }
    }

    public static void testPathfinding(Vector2D start, Vector2D end, PerlinGridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.YELLOW);
    }
}