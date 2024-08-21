package Structures;

import Tools.math;

import java.io.Serializable;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

public class Matrix implements Serializable {
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

    public Matrix(float[] data, int rows, int cols) {
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("Data length does not match the specified dimensions");
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = data[i * cols + j];
            }
        }
    }

    public static Matrix elementwiseSquare(Matrix matrix) {
        Matrix result = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                result.data[i][j] = matrix.data[i][j] * matrix.data[i][j];
            }
        }
        return result;
    }

    public static Matrix elementwiseSquareRoot(Matrix matrix) {
        Matrix result = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                result.data[i][j] = (float) Math.sqrt(matrix.data[i][j]);
            }
        }
        return result;
    }

    public static Matrix elementWiseDivide(Matrix matrix, Matrix divisor) {
        if (matrix.rows != divisor.rows || matrix.cols != divisor.cols) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for element-wise division.");
        }

        Matrix result = new Matrix(matrix.rows, matrix.cols);
        for (int i = 0; i < matrix.rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                if (divisor.data[i][j] == 0) {
                    throw new IllegalArgumentException("Division by zero encountered in matrix.");
                }
                result.data[i][j] = matrix.data[i][j] / divisor.data[i][j];
            }
        }

        return result;

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
    public void randomize(float min, float max) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = math.randomFloat(min, max);
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

    public static Matrix add(Matrix matrix, float v) {
        Matrix result = new Matrix(matrix.getRows(), matrix.getCols());
        for (int r = 0; r < matrix.getRows(); r++) {
            for (int c = 0; c < matrix.getCols(); c++) {
                result.set(c, r, matrix.get(c, r) + v);
            }
        }
        return result;
    }

    public void add(int row, int column, float value) {
        data[row][column] += value;
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

    public float sumOfSquares() {
        float sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return sum;
    }

    public void divide(float scalar) {
        if (scalar == 0) {
            throw new ArithmeticException("Cannot divide by zero");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] /= scalar;
            }
        }
    }

    public static Matrix divide(Matrix matrix, float scalar) {
        Matrix result = matrix.copy();
        result.divide(scalar);
        return result;
    }

    public Matrix transpose() {
        return transpose(this.copy());
    }

    public String dims() {
        return "[r:"+rows+", c:"+cols+"]";
    }

    public Matrix copy() {
        float[][] newData = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, newData[i], 0, cols);
        }
        return new Matrix(newData);
    }

    public static void copy(Matrix source, Matrix target) {
        if (source.rows != target.rows || source.cols != target.cols) {
            throw new IllegalArgumentException("Source and target matrices must have the same dimensions.");
        }
        for (int i = 0; i < source.rows; i++) {
            System.arraycopy(source.data[i], 0, target.data[i], 0, source.cols);
        }
    }

    public Matrix clip(float min, float max) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int r = 0; r < this.rows; r++) {
            for (int c = 0; c < this.cols; c++) {
                result.set(c, r, Math.max(min, Math.min(max, this.get(c, r))));
            }
        }
        return result;
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

    /*
    -----------------------------------------------------------------------------

    STATIC METHODS

    -----------------------------------------------------------------------------
     */

    public static Matrix add(Matrix a, Matrix b) {
        Matrix res = a.copy();
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("The matrices must have the same dimensions.");
        }

        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }

        return res;
    }

    public static Matrix subtract(Matrix a, Matrix b) {
        Matrix res = a.copy();
        if (a.rows != b.rows || a.cols != b.cols) {
            throw new IllegalArgumentException("The matrices must have the same dimensions.");
        }

        for (int i = 0; i < res.rows; i++) {
            for (int j = 0; j < res.cols; j++) {
                res.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }

        return res;
    }

    public static Matrix multiply(Matrix matrix, float value) {
        Matrix res = matrix.copy();
        res.multiply(value);
        return res;
    }

    public static Matrix transpose(Matrix matrix) {
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

    /*
    -----------------------------------------------------------------------------

    MULTIPLICATION

    -----------------------------------------------------------------------------
    */

    public static Matrix elementWiseMultiply(Matrix A, Matrix B) {
        if (A.rows != B.rows || A.cols != B.cols) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for element-wise multiplication. (A:"+A.dims()+" != B:"+B.dims()+")");
        }

        Matrix result = new Matrix(A.rows, A.cols);

        IntStream.range(0, A.rows).parallel().forEach(i -> {
            for (int j = 0; j < A.cols; j++) {
                result.data[i][j] = A.data[i][j] * B.data[i][j];
            }
        });

        return result;
    }

    public static Matrix multiply(Matrix A, Matrix B) {
        if (A.cols != B.rows) {
            System.out.println(B);
            throw new IllegalArgumentException("A's columns must match B's rows ("+A.cols+"!="+B.rows+") - A.dims="+A.dims()+", B.dims="+B.dims());
        }

        AtomicReference<float[][]> C = new AtomicReference<>(new float[A.rows][B.cols]);
        Matrix BT = transpose(B);

        POOL.invoke(new MultiplyTask(A, BT, C, 0, A.rows, 0, B.cols, 0, A.cols));

        return new Matrix(C.get());
    }

    public void multiply(Matrix B) {
        Matrix res = multiply(this, B);
        this.rows = res.rows;
        this.cols = res.cols;
        this.data = res.data;
    }

    public int getHeight() {
        return rows;
    }
    public int getRows() {
        return rows;
    }

    public int getWidth() {
        return cols;
    }
    public int getCols() {
        return cols;
    }

    public Matrix toRowMatrix() {
        Matrix result = new Matrix(1, rows*cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.set(r*cols+c, 0, get(c, r));
            }
        }
        return result;
    }

    public float getSum() {
        float sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    public float getMeanAverage() {
        return getSum() / (rows * cols);
    }

    private static class MultiplyTask extends RecursiveAction {
        private final Matrix A, BT;
        private final AtomicReference<float[][]> C;
        private final int rowStart, rowEnd, colStart, colEnd, depthStart, depthEnd;

        MultiplyTask(Matrix A, Matrix BT, AtomicReference<float[][]> C,
                     int rowStart, int rowEnd,
                     int colStart, int colEnd,
                     int depthStart, int depthEnd) {
            this.A = A;
            this.BT = BT;
            this.C = C;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
            this.colStart = colStart;
            this.colEnd = colEnd;
            this.depthStart = depthStart;
            this.depthEnd = depthEnd;
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
            float[][] localC = new float[rowEnd - rowStart][colEnd - colStart];
            for (int i0 = rowStart; i0 < rowEnd; i0 += TILE_SIZE) {
                for (int j0 = colStart; j0 < colEnd; j0 += TILE_SIZE) {
                    for (int k0 = depthStart; k0 < depthEnd; k0 += TILE_SIZE) {
                        multiplyTile(i0, j0, k0, localC);
                    }
                }
            }
            mergeResult(localC);
        }

        private void multiplyTile(int i0, int j0, int k0, float[][] localC) {
            int iMax = Math.min(i0 + TILE_SIZE, rowEnd);
            int jMax = Math.min(j0 + TILE_SIZE, colEnd);
            int kMax = Math.min(k0 + TILE_SIZE, depthEnd);

            for (int i = i0; i < iMax; i++) {
                for (int j = j0; j < jMax; j += UNROLL_FACTOR) {
                    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

                    for (int k = k0; k < kMax; k++) {
                        float aik = A.data[i][k];
                        float product = aik * BT.data[j][k];
                        sum0 += product;

                        if (product > Float.MAX_VALUE) {
                            throw new ArithmeticException("Overflow during matrix multiplication at (" + i + ", " + j + ", " + k + ")");
                        }
                        if (product < Float.MIN_VALUE) {
                            throw new ArithmeticException("Underflow during matrix multiplication at (" + i + ", " + j + ", " + k + ")");
                        }

                        if (j + 1 < jMax) sum1 += aik * BT.data[j + 1][k];
                        if (j + 2 < jMax) sum2 += aik * BT.data[j + 2][k];
                        if (j + 3 < jMax) sum3 += aik * BT.data[j + 3][k];
                    }

                    if (sum0 > Float.MAX_VALUE || sum1 > Float.MAX_VALUE || sum2 > Float.MAX_VALUE || sum3 > Float.MAX_VALUE) {
                        throw new ArithmeticException("Overflow during matrix multiplication at (" + i + ", " + j + ")");
                    }
                    if (sum0 < Float.MIN_VALUE || sum1 < Float.MIN_VALUE || sum2 < Float.MIN_VALUE || sum3 < Float.MIN_VALUE) {
                        throw new ArithmeticException("Underflow during matrix multiplication at (" + i + ", " + j + ")");
                    }

                    localC[i - rowStart][j - colStart] += sum0;
                    if (j + 1 < jMax) localC[i - rowStart][j + 1 - colStart] += sum1;
                    if (j + 2 < jMax) localC[i - rowStart][j + 2 - colStart] += sum2;
                    if (j + 3 < jMax) localC[i - rowStart][j + 3 - colStart] += sum3;
                }
            }
        }

        private void mergeResult(float[][] localC) {
            float[][] globalC = C.get();
            synchronized (C) {
                for (int i = 0; i < localC.length; i++) {
                    for (int j = 0; j < localC[0].length; j++) {
                        globalC[i + rowStart][j + colStart] += localC[i][j];
                    }
                }
            }
        }
    }
}
