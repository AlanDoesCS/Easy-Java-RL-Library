package Structures;

import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public class Matrix {
    private float[][] data;
    int rows, cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new float[rows][cols];
    }

    public Matrix(float[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
    }

    public void fill(float value) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }

    public void randomize() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) (Math.random() * 2 - 1);
            }
        }
    }

    public void add(float n) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += n;
            }
        }
    }

    public void add(Matrix m) {
        if (rows != m.rows || cols != m.cols) {
            throw new IllegalArgumentException("The matrices must have the same dimensions.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += m.data[i][j];
            }
        }
    }

    public void subtract(float n) {
        add(-n);
    }

    public void subtract(Matrix m) {
        if (rows != m.rows || cols != m.cols) {
            throw new IllegalArgumentException("The matrices must have the same dimensions.");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= m.data[i][j];
            }
        }
    }

    public void multiply(float n) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= n;
            }
        }
    }

    public Matrix transpose() {
        float[][] newData = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newData[j][i] = data[i][j];
            }
        }
        data = newData;
        int temp = rows;
        rows = cols;
        cols = temp;
        return this;
    }

    public static Matrix getIdentityMatrix(int width) {
        float[][] data = new float[width][width];
        for (int i = 0; i < width; i++) {
            data[i][i] = 1;
        }
        return new Matrix(data);
    }
    public static Matrix getIdentityMatrix(Matrix m) {
        assert (m.rows == m.cols); // Make sure matrix m is a square matrix
        return getIdentityMatrix(m.cols);
    }

    //                                          1           2
    public static Matrix getNumberedMatrix(int width, int height) {
        Matrix m = new Matrix(height, width);
        //                      1       2

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                m.data[j][i] = width*j + i + 1;
            }
        }

        return m;
    }

    public static Matrix transpose(Matrix B) {
        return B.transpose();
    }

    // Slower Matrix multiplication
    @Deprecated
    public static Matrix multiply(Matrix A, Matrix B) {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("Invalid matrix dimensions");
        }

        Matrix result = new Matrix(A.rows, B.cols);
        for (int i = 0; i < result.rows; i++) {
            for (int j = 0; j < result.cols; j++) {
                for (int k = 0; k < A.cols; k++) {
                    result.data[i][j] += A.data[i][k] * B.data[k][j];
                }
            }
        }
        return result;
    }

    /*
    Faster matrix multiplication algorithm, written after watching this video:
    https://www.youtube.com/watch?v=QGYvbsHDPxo (Please go watch, it's really great)

    Also yes, it is faster for large matrices than an approach with fewer loops
    ---

    Steps:
    - Validate input
    - Transpose matrix B for improved cache efficiency
    - Outer loops to create blocks (i, j, k)
    - Compute values for inner blocks, using threaded approach
    - return resulting Matrix
     */

    public static Matrix multiply(Matrix A, Matrix B, int blockSize)  {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("Invalid matrix dimensions: Cannot multiply "+A.dims()+", "+B.dims()+": (A.c) "+A.cols+"!= (B.r)"+B.rows);
        }

        Matrix result = new Matrix(A.rows, B.cols);
        B.transpose();  // Transpose matrix B for better cache efficiency

        // executor for threads
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        for (int i = 0; i < A.rows; i += blockSize) {
            for (int j = 0; j < B.cols; j += blockSize) {
                for (int k = 0; k < A.cols; k += blockSize) {
                    final int iStart = i;
                    final int jStart = j;
                    final int kStart = k;

                    executor.execute(() -> {
                        for (int ii = iStart; ii < Math.min(iStart + blockSize, A.rows); ii++) {
                            for (int jj = jStart; jj < Math.min(jStart + blockSize, B.cols); jj++) {
                                float sum = 0;
                                for (int kk = kStart; kk < Math.min(kStart + blockSize, A.cols); kk++) {
                                    System.out.println(ii + ", " + jj + ", " + kk);
                                    sum += A.data[ii][kk] * B.data[jj][kk];
                                }
                                synchronized (result) {
                                    result.data[ii][jj] += sum;
                                }
                            }
                        }
                    });
                }
            }
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            // Restore the interrupted status
            Thread.currentThread().interrupt();
            throw new RuntimeException("Matrix multiplication was interrupted", e);
        }


        return result;
    }

    private String dims() {
        return "[r:"+rows+", c:"+cols+"]";
    }


    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        final int dp = 3;
        int multiplier = (int) Math.pow(10, dp);
        for (int i = 0; i < rows; i++) {
            sb.append("\n[");
            for (int j = 0; j < cols; j++) {
                float roundedVal = (float) Math.round(data[i][j]*multiplier)/multiplier;
                sb.append(roundedVal);
                if (j < cols - 1) {
                    sb.append(",\t");
                }
            }
            sb.append("]");
            if (i < rows - 1) {
                sb.append(",");
            }
        }
        sb.append("\n]");
        return sb.toString();
    }

    public float get(int x, int y) {
        return data[y][x];
    }

    public void set(int x, int y, float value) {
        data[y][x] = value;
    }
}
