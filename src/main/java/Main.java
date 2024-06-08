import Structures.DQN;
import Structures.Layer;
import Structures.Matrix;
import Tools.Perlin2D;
import Tools.PerlinNoise_Visualiser;
import Training.ActivationFunction;
import Training.RandomGridMesh;
import Training.Sigmoid;
import Tools.Perlin1D;

import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        Random rand = new Random();

        int width=10, height=10;
        RandomGridMesh environment = new RandomGridMesh(width, height);
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

        // System.out.println(environment);

        ActivationFunction sig = new Sigmoid();
        List<Layer> layers = List.of(
                new Layer(numSquares, 7, sig, 0),
                new Layer(7, 8, sig, 0),
                new Layer(8, 5, sig, 0)
        );

        DQN net = new DQN(numSquares, layers, 4, 1, sig, 0);

        // System.out.println(net.getOutput(input));

        int octaves = 8;
        float persistence = 0.3f;
        Perlin1D p1 = new Perlin1D(octaves, persistence);
        Perlin2D p2 = new Perlin2D(octaves, persistence);

        new PerlinNoise_Visualiser(p2, 0.01f);

        // new DQN_Visualiser(net);
    }
}
