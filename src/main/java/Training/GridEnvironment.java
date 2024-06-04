package Training;

import Structures.Matrix;

public abstract class GridEnvironment {
    public int width, height;
    private Matrix gridMatrix;

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        gridMatrix = new Matrix(width, height);
    }

    public int getNumSquares() {
        return width * height;
    }

    public abstract void fill();

    public float get(int x, int y) {
        return gridMatrix.get(x, y);
    }
    public float get(int i) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        return get(x, y);
    }

    public void set(int x, int y, float value) {
        gridMatrix.set(x, y, value);
    }
    public void set(int i, float value) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        set(x, y, value);
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
