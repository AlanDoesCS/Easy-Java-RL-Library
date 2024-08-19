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

    public int getCurrentSteps() {
        return currentSteps;
    }

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2D(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
        this.gridMatrix = new Matrix(height, width);
        this.stateTensor = new Tensor(3, height, width); // Environment, Agent, Goal channels

        this.maxSteps = width*height;
        this.currentSteps = 0;

        this.minReward = -1; // Worst case: DNF
        this.maxReward = getCompletionReward() + 1 + getValidMoveReward(); // Best case: reach goal + max step reward + valid move reward
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
        return math.clamp(gridMatrix.get(x, y), 0, 1);
    }
    public float get(int i) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        return get(x, y);
    }

    public Tensor getState() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                stateTensor.set(0, y, x, get(x, y));  // Environment
                stateTensor.set(1, y, x, (x == agentPosition.getX() && y == agentPosition.getY()) ? 1 : 0);  // Agent
                stateTensor.set(2, y, x, (x == goalPosition.getX() && y == goalPosition.getY()) ? 1 : 0);  // Goal
            }
        }
        return stateTensor.copy();
    }

    float getStepReward(Vector2D oldPosition, Vector2D newPosition) {
        float oldDistance = oldPosition.manhattanDistanceTo(goalPosition);
        float newDistance = newPosition.manhattanDistanceTo(goalPosition);

        return oldDistance == 0 ? 0f : (oldDistance - newDistance) / oldDistance;
    }

    public MoveResult step(int action) {
        float reward = 0;
        currentSteps++;
        Vector2D oldPosition = getAgentPosition();
        Vector2D newPosition = oldPosition.copy();

        getNewPosFromAction(action, newPosition);
        boolean validMove = isValidPositionInBounds((int) newPosition.getX(), (int) newPosition.getY());
        boolean maxStepsReached = currentSteps >= maxSteps;
        boolean done = false;

        if (validMove) {
            setAgentPosition(newPosition);
            reward += getStepReward(oldPosition, newPosition);
            reward += getValidMoveReward(); // encourage valid moves

            done = newPosition.equals(getGoalPosition());

            if (done) {
                reward += getCompletionReward();
            } else if (!maxStepsReached) {
                reward -= (float) (get((int)newPosition.getX(), (int)newPosition.getY())*0.5);
            }
        } else {
            reward -= getInvalidMovePunishment(); // discourage invalid moves
        }

        if (maxStepsReached && !done) {
            // Don't punish DNF because it introduces noise, and the Agent has no representation of time - instead encourage completion as that has no time constraints
            done = true;
        }

        reward = math.clamp(math.scale(reward, minReward, maxReward, -1, 1), -1, 1);
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
        return 5;
    }

    float getValidMoveReward() {
        return 0.8f;
    }

    float getDNFPunishment() {
        return 3;
    }

    float getInvalidMovePunishment() {
        return 0.5f;
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
