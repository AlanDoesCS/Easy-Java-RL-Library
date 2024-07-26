import Structures.DQN;
import Structures.Layer;
import Structures.Matrix;
import Structures.Vector2D;
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
        final int octaves = 8;
        final float persistence = 0.9f;
        final float step = 0.01f;

        PerlinGridEnvironment environment = new PerlinGridEnvironment(width, height, octaves, persistence, step);
        int numSquares = environment.getNumSquares();

        int startX=rand.nextInt(width+1), startY=rand.nextInt(height+1);
        int endX=rand.nextInt(width+1), endY=rand.nextInt(height+1);

        float[][] inputArr = new float[numSquares+4][1]; // Takes all square weights and start/end positions as parameters
        for (int i = 0; i < numSquares; i++) {
            inputArr[i][0] = environment.get(i);
        }
        // Start Position:
        inputArr[numSquares][0] = startX;
        inputArr[numSquares+1][0] = startY;

        // End Position:
        inputArr[numSquares+2][0] = endX;
        inputArr[numSquares+3][0] = endY;

        Matrix input = new Matrix(inputArr);

        System.out.println(input);

        // testPathfinding(environment.getRandomCoordinateInBounds(), environment.getRandomCoordinateInBounds(), environment);

        Matrix testInput = new Matrix(new float[][]{
                {0.2f},
                {-0.2f},
                {0.6f},
                {-0.4f},
                {0.1f},
                {0.9f}
        });

        ActivationFunction sig = new Sigmoid();
        List<Layer> layers = List.of(
                new Layer(input.getHeight(), 7, sig, 0),
                new Layer(7, 8, sig, 0),
                new Layer(8, 5, sig, 0)
        );

        DQN net = new DQN(numSquares, layers, 4, 1, sig, 0);

        for (int i = 0; i < 5; i++) {
            System.out.println("Run " + (i+1) + ":");
            Matrix output = net.getOutput(input);
            System.out.println(output);
            System.out.println();
        }


        // new DQN_Visualiser(net);
    }

    public static void testPathfinding(Vector2D start, Vector2D end, PerlinGridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, Color.YELLOW);
    }
}