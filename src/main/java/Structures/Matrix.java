package Structures;

import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;

public class Matrix {
    private static final int TILE_SIZE = 32;
    private static final int UNROLL_FACTOR = 4;
    private static final int PARALLELISM_THRESHOLD = 1024;
    private static final ForkJoinPool POOL = ForkJoinPool.commonPool();

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

    private static Matrix transpose(Matrix matrix) {
        int rows = matrix.data.length;
        int cols = matrix.data[0].length;
        Matrix transposed = new Matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j][i] = matrix.data[i][j];
            }
        }

        return transposed;
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

    private static class MultiplyTask extends RecursiveAction {
        private Matrix A, BT, C;
        private int rowStart, rowEnd, colStart, colEnd, depthStart, depthEnd;

        MultiplyTask(Matrix A, Matrix BT, Matrix C,
                     int rowStart, int rowEnd,
                     int colStart, int colEnd,
                     int depthStart, int depthEnd) {
            this.A = A; this.BT = BT; this.C = C;
            this.rowStart = rowStart; this.rowEnd = rowEnd;
            this.colStart = colStart; this.colEnd = colEnd;
            this.depthStart = depthStart; this.depthEnd = depthEnd;
        }

        @Override
        protected void compute() {
            int rowSize = rowEnd - rowStart;
            int colSize = colEnd - colStart;
            int depthSize = depthEnd - depthStart;

            if (rowSize * colSize * depthSize <= PARALLELISM_THRESHOLD) {
                multiplySequential();
                return;
            }

            if (rowSize >= colSize && rowSize >= depthSize) {
                int mid = rowStart + rowSize / 2;
                invokeAll(
                        new MultiplyTask(A, BT, C, rowStart, mid, colStart, colEnd, depthStart, depthEnd),
                        new MultiplyTask(A, BT, C, mid, rowEnd, colStart, colEnd, depthStart, depthEnd)
                );
            } else if (colSize >= depthSize) {
                int mid = colStart + colSize / 2;
                invokeAll(
                        new MultiplyTask(A, BT, C, rowStart, rowEnd, colStart, mid, depthStart, depthEnd),
                        new MultiplyTask(A, BT, C, rowStart, rowEnd, mid, colEnd, depthStart, depthEnd)
                );
            } else {
                int mid = depthStart + depthSize / 2;
                invokeAll(
                        new MultiplyTask(A, BT, C, rowStart, rowEnd, colStart, colEnd, depthStart, mid),
                        new MultiplyTask(A, BT, C, rowStart, rowEnd, colStart, colEnd, mid, depthEnd)
                );
            }
        }

        private void multiplySequential() {
            for (int i0 = rowStart; i0 < rowEnd; i0 += TILE_SIZE) {
                for (int j0 = colStart; j0 < colEnd; j0 += TILE_SIZE) {
                    for (int k0 = depthStart; k0 < depthEnd; k0 += TILE_SIZE) {
                        multiplyTile(i0, j0, k0);
                    }
                }
            }
        }

        private void multiplyTile(int i0, int j0, int k0) {
            int iMax = Math.min(i0 + TILE_SIZE, rowEnd);
            int jMax = Math.min(j0 + TILE_SIZE, colEnd);
            int kMax = Math.min(k0 + TILE_SIZE, depthEnd);

            for (int i = i0; i < iMax; i++) {
                for (int j = j0; j < jMax; j += UNROLL_FACTOR) {
                    float sum0 = C.data[i][j];
                    float sum1 = j + 1 < jMax ? C.data[i][j + 1] : 0;
                    float sum2 = j + 2 < jMax ? C.data[i][j + 2] : 0;
                    float sum3 = j + 3 < jMax ? C.data[i][j + 3] : 0;

                    for (int k = k0; k < kMax; k++) {
                        float aik = A.data[i][k];
                        sum0 += aik * BT.data[j][k];
                        if (j + 1 < jMax) sum1 += aik * BT.data[j + 1][k];
                        if (j + 2 < jMax) sum2 += aik * BT.data[j + 2][k];
                        if (j + 3 < jMax) sum3 += aik * BT.data[j + 3][k];
                    }

                    C.data[i][j] = sum0;
                    if (j + 1 < jMax) C.data[i][j + 1] = sum1;
                    if (j + 2 < jMax) C.data[i][j + 2] = sum2;
                    if (j + 3 < jMax) C.data[i][j + 3] = sum3;
                }
            }
        }
    }

    @Deprecated
    public static Matrix multiply_deprecated(Matrix A, Matrix B) {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("A's columns must match B's rows");
        }

        int rowsA = A.rows;
        int colsB = B.cols;
        int colsA = A.cols;

        Matrix C = new Matrix(rowsA, colsB);

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                float sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum += A.data[i][k] * B.data[k][j];
                }
                C.data[i][j] = sum;
            }
        }

        return C;
    }

    public static Matrix multiply(Matrix A, Matrix B) {
        if (A.cols != B.rows) {
            throw new IllegalArgumentException("A's columns must match B's rows");
        }

        Matrix C = new Matrix(A.rows, B.cols);
        Matrix BT = transpose(B);

        POOL.invoke(new MultiplyTask(A, BT, C, 0, A.rows, 0, B.cols, 0, A.cols));

        return C;
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
