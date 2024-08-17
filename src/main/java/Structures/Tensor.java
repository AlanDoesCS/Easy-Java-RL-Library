package Structures;

public class Tensor {
    private final int depth, height, width;
    float[][][] data;

    public Tensor(int depth, int height, int width) {
        this.data = new float[depth][height][width];
        this.depth = depth;
        this.height = height;
        this.width = width;
    }

    public Tensor(float[][][] data) {
        this.data = data;
        this.depth = data.length;
        this.height = data[0].length;
        this.width = data[0][0].length;
    }

    // Accessors and Mutators
    public float get(int d, int h, int w) {
        return data[d][h][w];
    }
    public void set(int d, int h, int w, float value) {
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

    public float[][][] getData() {
        return data;
    }

    public void setData(float[][][] data) {
        this.data = data;
    }

    public Tensor copy() {
        float[][][] copiedData = new float[depth][height][width];
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                System.arraycopy(data[d][h], 0, copiedData[d][h], 0, width);
            }
        }
        return new Tensor(copiedData);
    }
}
