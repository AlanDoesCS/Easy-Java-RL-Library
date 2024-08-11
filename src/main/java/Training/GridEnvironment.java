package Training;

import Structures.Matrix;
import Structures.Tensor;
import Structures.Vector2D;
import Tools.math;

public abstract class GridEnvironment extends Environment {
    public int width, height;
    private Matrix gridMatrix;
    private Tensor stateTensor;
    private Vector2D startPosition;
    private Vector2D agentPosition;
    private Vector2D goalPosition;

    protected int maxSteps, currentSteps;

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2D(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
        this.gridMatrix = new Matrix(height, width);
        this.stateTensor = new Tensor(3, height, width); // Environment, Agent, Goal channels


        this.maxSteps = width*height*2;
        this.currentSteps = 0;
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
        this.currentSteps = 0;
    }

    public float get(int x, int y) {
        return gridMatrix.get(x, y);
    }
    public float get(int i) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        return get(x, y);
    }

    public Tensor getState() {
        Tensor state = new Tensor(3, height, width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                state.set(x, y, 0, get(x, y));  // Environment
                state.set(x, y, 1, (x == agentPosition.getX() && y == agentPosition.getY()) ? 1 : 0);  // Agent
                state.set(x, y, 2, (x == goalPosition.getX() && y == goalPosition.getY()) ? 1 : 0);  // Goal
            }
        }
        return state;
    }

    public MoveResult step(int action) {
        Vector2D currentPosition = getAgentPosition();
        Vector2D newPosition = currentPosition.copy();

        PerlinGridEnvironment.getNewPosFromAction(action, newPosition);

        if (isValidPositionInBounds((int) newPosition.getX(), (int) newPosition.getY())) {
            setAgentPosition(newPosition);
            currentSteps++;
        } else {
            newPosition = currentPosition;
        }

        boolean done = newPosition.equals(getGoalPosition());
        boolean maxStepsReached = currentSteps >= maxSteps;

        if (maxStepsReached && !done) {
            done = true;
            return new MoveResult(getState(), -getDNFPunishment(), done);
        }

        float reward = done ? getCompletionReward() : -get((int)newPosition.getX(), (int)newPosition.getY());

        return new MoveResult(getState(), reward, done);
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

    boolean isValidPositionInBounds(int x, int y) { // by default, any position in bounds is valid
        return isInBounds(x, y);
    }

    public Vector2D getRandomCoordinateInBounds() {
        return new Vector2D(math.randomInt(0, width-1), math.randomInt(0, height-1));
    }

    /**
     * Updates the given position based on the specified action.
     *
     * @param action The action to be taken. The action is represented as an integer:
     *      0 - Move up
     *      1 - Move right
     *      2 - Move down
     *      3 - Move left
     *      4 - Do nothing
     * @param newPosition The position to be updated. This is a Vector2D object that will be modified
     *                    based on the action.
     */
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
        return math.fastSqrt(getStateSpace()-(width+height));
    }

    float getDNFPunishment() {
        return (math.fastSqrt(getStateSpace()-(width+height)))/3;
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
