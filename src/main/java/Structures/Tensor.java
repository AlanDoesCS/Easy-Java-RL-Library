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
    public float get(int x, int y, int z) {
        return data[z][y][x];
    }
    public void set(int x, int y, int z, float value) {
        data[z][y][x]=value;
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
}
