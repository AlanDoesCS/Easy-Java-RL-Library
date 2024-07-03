package Tools.Benchmarking;

import Structures.Matrix;

public class MatrixMultiplicationTimer extends MethodTimer {
    static Matrix A = Matrix.getIdentityMatrix(1024);
    static Matrix B = Matrix.getNumberedMatrix(1024, 1024);

    public long time1024BasicMultiplication() {
        long start = System.nanoTime();
        Matrix.multiply(A, B);
        long end = System.nanoTime();

        return end - start;
    }

    public long time1024WithBlocks(int blocksize) { // Timer for blocking with threads
        long start = System.nanoTime();

        Matrix.multiply(A, B, blocksize);

        long end = System.nanoTime();

        return end - start;
    }

    public long time1024WithBlocksRandomized(int blockSize) { // Timer for blocking with threads
        Matrix A = new Matrix(1024, 1024), B = new Matrix(1024, 1024);
        A.randomize(); B.randomize();

        long start = System.nanoTime();
        Matrix.multiply(A, B, blockSize);
        long end = System.nanoTime();

        return end - start;
    }

    public static long timeWithBlocks(Matrix A, Matrix B, int blockSize) { // Timer for blocking with threads
        long start = System.nanoTime();
        Matrix.multiply(A, B, blockSize);
        long end = System.nanoTime();

        return end - start;
    }

    public int findIdealBlockSize(int matrixWidth, int min, int max, int samples) {
        int idealBlockSize = min;
        long shortestTime = Long.MAX_VALUE;

        final Matrix A = Matrix.getIdentityMatrix(matrixWidth);
        final Matrix B = Matrix.getNumberedMatrix(matrixWidth, matrixWidth);

        for (int w = min; w<=max; w++) {
            long total = 0;
            for (int i=0; i<samples; i++) {
                total += timeWithBlocks(A, B, w);
            }
            long averageTime = total/samples;
            //System.out.println(w + " block width, time: " + averageTime/1e6 + "ms");
            System.out.println(w + "\t" + averageTime  + "\t" + averageTime/1e6);

            if (averageTime < shortestTime) {
                shortestTime = averageTime;
                idealBlockSize = w;

                //System.out.println("New ideal block size: " + w + " took on average: " + averageTime/1e6 + "ms");
            }
        }

        return idealBlockSize;
    }
}