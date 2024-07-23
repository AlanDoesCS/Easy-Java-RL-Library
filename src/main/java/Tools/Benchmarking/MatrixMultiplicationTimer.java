package Tools.Benchmarking;

import Structures.Matrix;

public class MatrixMultiplicationTimer extends MethodTimer {
    static Matrix A = Matrix.getIdentityMatrix(1024);
    static Matrix B = Matrix.getNumberedMatrix(1024, 1024);

    public static long time1024BasicMultiplication() {
        long start = System.nanoTime();
        Matrix.multiply_deprecated(A, B);
        long end = System.nanoTime();

        return end - start;
    }

    public static long time1024Fast() {
        long start = System.nanoTime();
        Matrix.multiply(A, B);
        long end = System.nanoTime();

        return end - start;
    }
}