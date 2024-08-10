package Training;

import Structures.Matrix;
import Structures.Vector2D;
import Tools.math;

public abstract class GridEnvironment extends Environment {
    public int width, height;
    private Matrix gridMatrix, stateMatrix;
    private Vector2D startPosition;
    private Vector2D agentPosition;
    private Vector2D goalPosition;

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2D(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
        this.gridMatrix = new Matrix(height, width);
        this.stateMatrix = new Matrix(getNumSquares()+width+height, 1); // width+height for one hot encoding of agent position
    }

    public int getNumSquares() {
        return width * height;
    }
    public int getWidth() { return width; }
    public int getHeight() { return height; }

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
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2D(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
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
                stateMatrix.set(0, i++, get(x, y));
            }
        }
        // position X
        int startOffset = width*height;
        encodeOneHot(stateMatrix, startOffset, width, (int) getAgentPosition().getX());

        startOffset += width;

        // position Y
        encodeOneHot(stateMatrix, startOffset, height, (int) getAgentPosition().getY());
        return stateMatrix;
    }

    private void encodeOneHot(Matrix columnMatrixTarget, int startOffset, int length, int... trueBitIndices) {
        for (int i = startOffset; i < startOffset+length; i++) {
            columnMatrixTarget.set(0, i, 0);
        }
        for (int trueBitIndex : trueBitIndices) {
            columnMatrixTarget.set(0, startOffset+trueBitIndex, 1);
        }
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
    public void setStartPosition(int x, int y) {
        this.startPosition.set(x, y);
    }
    public void setAgentPosition(Vector2D position) {
        this.agentPosition = position;
    }
    public void setGoalPosition(Vector2D position) {
        this.goalPosition = position;
    }
    public void setStartPosition(Vector2D position) {
        this.startPosition = position;
    }

    public Vector2D getAgentPosition() {
        return agentPosition;
    }
    public Vector2D getGoalPosition() {
        return goalPosition;
    }
    public Vector2D getStartPosition() {
        return startPosition;
    }

    public boolean isInBounds(int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    public Vector2D getRandomCoordinateInBounds() {
        return new Vector2D(math.randomInt(0, width-1), math.randomInt(0, height-1));
    }

    static void getNewPosFromAction(int action, Vector2D newPosition) {
        switch(action) {
            case 0: newPosition.addJ(-1); break; // Move up
            case 1: newPosition.addI(1); break;  // Move right
            case 2: newPosition.addJ(1); break;  // Move down
            case 3: newPosition.addI(-1); break; // Move left
            case 4: break; // Do nothing
        }
    }

    float getCompletionReward() {
        return math.fastSqrt(getStateSpace()-4);
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
