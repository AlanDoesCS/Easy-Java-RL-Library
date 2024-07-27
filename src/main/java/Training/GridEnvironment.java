package Training;

import Structures.Matrix;
import Structures.Vector2D;
import Tools.math;

public abstract class GridEnvironment extends Environment {
    public static class MoveResult {
        public Matrix state;
        public float reward;
        public boolean done;

        public MoveResult(Matrix state, float reward, boolean done) {
            this.state = state;
            this.reward = reward;
            this.done = done;
        }
    }

    public int width, height;
    private Matrix gridMatrix, stateMatrix;
    private Vector2D agentPosition, goalPosition;

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        this.agentPosition = getRandomCoordinateInBounds();
        this.goalPosition = getRandomCoordinateInBounds();
        this.gridMatrix = new Matrix(width, height);
        this.stateMatrix = new Matrix(getNumSquares()+4, 1);
    }

    public int getNumSquares() {
        return width * height;
    }
    public int getStateSpace() {return stateMatrix.getHeight();}
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public Vector2D getAgentPosition() {
        return agentPosition;
    }
    public Vector2D getGoalPosition() {
        return goalPosition;
    }
    public Matrix getGridMatrix() {
        return gridMatrix;
    }

    // fill matrix
    abstract void fill();
    public void refill() {
        fill();
    }

    public void randomize() {
        refill();
    }

    public float get(int x, int y) {
        return gridMatrix.get(x, y);
    }
    public float get(int i) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        return get(x, y);
    }

    public Matrix getState() {
        int i=0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                stateMatrix.set(0, i++, get(x,y));
            }
        }
        stateMatrix.set(0, i++, agentPosition.getI());
        stateMatrix.set(0, i++, agentPosition.getJ());

        // End Position:
        stateMatrix.set(0, i++, goalPosition.getI());
        stateMatrix.set(0, i, goalPosition.getJ());
        return stateMatrix;
    }

    public void set(int x, int y, float value) {
        gridMatrix.set(x, y, value);
    }
    public void set(int i, float value ) { // simplifies process for creating the environment
        int x = i % width;
        int y = i / width;
        set(x, y, value);
    }

    public void setAgentPosition(int x, int y) {
        this.agentPosition.set(x, y);
    }
    public void setGoalPosition(int x, int y) {
        this.goalPosition.set(x, y);
    }
    public void setAgentPosition(Vector2D position) {
        this.agentPosition = position;
    }
    public void setGoalPosition(Vector2D position) {
        this.goalPosition = position;
    }

    public boolean isInBounds(int x, int y) {
        return x > 0 && x < width - 1 && y > 0 && y < height - 1;
    }

    public Vector2D getRandomCoordinateInBounds() {
        return new Vector2D(math.randomInt(0, width-1), math.randomInt(0, height-1));
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
