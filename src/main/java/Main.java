import Structures.FNN;
import Structures.Matrix;

public class Main {
    public static void main(String[] args) {
        FNN net = new FNN(3, 1, 2);
        Matrix input = new Matrix(
                new float[][]
                        {
                                {2f},
                                {1f},
                                {0f}
                        }
        );

        System.out.println(net.getOutput(input));
    }
}
