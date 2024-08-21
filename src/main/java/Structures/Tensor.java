package Structures;

public class Tensor {
    private final int depth, height, width;
    double[][][] data;

    public Tensor(int depth, int height, int width) {
        this.data = new double[depth][height][width];
        this.depth = depth;
        this.height = height;
        this.width = width;
    }

    public Tensor(double[][][] data) {
        this.data = data;
        this.depth = data.length;
        this.height = data[0].length;
        this.width = data[0][0].length;
    }

    // Accessors and Mutators
    public double get(int d, int h, int w) {
        return data[d][h][w];
    }
    public void set(int d, int h, int w, double value) {
        data[d][h][w]=value;
    }

    public int getDepth() {
        return depth;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public double[][][] getData() {
        return data;
    }

    public void setData(double[][][] data) {
        this.data = data;
    }

    public Tensor copy() {
        double[][][] copiedData = new double[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(data[d][h], 0, copiedData[d][h], 0, width);
            }
        }
        return new Tensor(copiedData);
    }

    public double getSum() {
        double sum = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    sum += data[d][h][w];
                }
            }
        }
        return sum;
    }
}
