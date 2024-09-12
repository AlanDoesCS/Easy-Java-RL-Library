package Training.Environments;

import Structures.MatrixDouble;
import Structures.Tensor;
import Structures.Vector2;
import Tools.math;

public abstract class GridEnvironment extends Environment {
    public int width, height;
    private MatrixDouble gridMatrix;
    private Vector2 startPosition;
    private Vector2 agentPosition;
    private Vector2 goalPosition;

    protected int maxSteps, currentSteps;

    public int getCurrentSteps() {
        return currentSteps;
    }

    public GridEnvironment(int width, int height) {
        this.width = width;
        this.height = height;
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
        this.gridMatrix = new MatrixDouble(height, width);

        this.maxSteps = width * height;
        this.currentSteps = 0;

        this.minReward = -1;
        this.maxReward = getValidMoveReward() + 1; // Best case: valid move + moved to goal
    }

    public int getNumSquares() {
        return width * height;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public MatrixDouble getGridMatrix() {
        return gridMatrix;
    }

    // Fill matrix
    abstract void fill();

    public void refill() {
        fill();
    }

    public void randomize() {
        refill();
        this.agentPosition = getRandomCoordinateInBounds();
        this.startPosition = new Vector2(agentPosition);
        this.goalPosition = getRandomCoordinateInBounds();
        this.currentSteps = 0;
    }

    public double get(int x, int y) {
        return math.clamp(gridMatrix.get(x, y), 0, 1);
    }

    public double get(int i) { // simplifies process getting cells for Neural Net
        int x = i % width;
        int y = i / width;
        return get(x, y);
    }

    public Tensor getStateTensor() {
        Tensor stateTensor = new Tensor(3, height, width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                stateTensor.set(0, y, x, get(x, y));  // Environment
                stateTensor.set(1, y, x, (x == agentPosition.getX() && y == agentPosition.getY()) ? 1 : 0);  // Agent
                stateTensor.set(2, y, x, (x == goalPosition.getX() && y == goalPosition.getY()) ? 1 : 0);  // Goal
            }
        }
        return stateTensor;
    }

    /**
     * Converts the current state of the grid environment into a column matrix.
     * The matrix includes the grid values, agent position, and goal position.
     * <p>- This is useful for training MLP layers in a neural network.
     * @return A Column MatrixDouble object representing the state of the grid environment as well as the agent and goal positions.
     */
    public MatrixDouble getStateAsColumnMatrix() {
        MatrixDouble state = new MatrixDouble(getGridWidth() * getGridHeight() + 4, 1); // +4 for agent position and goal position
        int i = 0;
        for (int y = 0; y < getGridHeight(); y++) {
            for (int x = 0; x < getGridWidth(); x++) {
                state.set(0, i++, get(x, y));
            }
        }
        Vector2 agentPos = Vector2.normalise(getAgentPosition(), getGridWidth() - 1, getGridHeight() - 1);
        Vector2 goalPos = Vector2.normalise(getGoalPosition(), getGridWidth() - 1, getGridHeight() - 1);
        state.set(0, i++, agentPos.getX());
        state.set(0, i++, agentPos.getY());
        state.set(0, i++, goalPos.getX());
        state.set(0, i, goalPos.getY());

        return state;
    }

    public Object getState() {
        switch (Environment.stateType) {
            case PositionVectorOnly:
                Vector2 agentNorm = Vector2.normalise(getAgentPosition(), getGridWidth() - 1, getGridHeight() - 1);
                Vector2 goalNorm = Vector2.normalise(getGoalPosition(), getGridWidth() - 1, getGridHeight() - 1);
                return new MatrixDouble(new double[][]{
                        {agentNorm.getX(), agentNorm.getY(), goalNorm.getX(), goalNorm.getY()}
                }).toColumnMatrix();
            case PositionAndGridAsColumn:
                return getStateAsColumnMatrix();
            case PositionAndGridAsLayers:
                return getStateTensor();
            default:
                return null;
        }
    }

    double getStepReward(Vector2 oldPosition, Vector2 newPosition) {
        double oldDistance = oldPosition.manhattanDistanceTo(goalPosition);
        double newDistance = newPosition.manhattanDistanceTo(goalPosition);

        // Prevent division by zero in case oldDistance is 0
        if (oldDistance == 0) {
            return 0; // No reward if the agent is already at the goal
        }

        // Provide intermediate reward for moving closer to the goal
        double progressReward = (oldDistance - newDistance) / oldDistance;

        if (progressReward > 0) {
            // Reward for moving closer
            return progressReward * 0.8;  // Scale this factor as needed
        } else {
            // Penalty for moving away
            return -0.2;
        }
    }

    public MoveResult step(int action) {
        float reward = 0;
        currentSteps++;
        Vector2 oldPosition = getAgentPosition();
        Vector2 newPosition = oldPosition.copy();

        // Determine the new position based on the chosen action
        getNewPosFromAction(action, newPosition);
        boolean validMove = isValidPositionInBounds((int) newPosition.getX(), (int) newPosition.getY());
        boolean maxStepsReached = currentSteps >= maxSteps;
        boolean done = false;

        if (validMove) {
            // Update agent's position
            setAgentPosition(newPosition);

            // Add the scaled reward based on proximity to the goal
            reward += (float) getStepReward(oldPosition, newPosition);

            // Encourage valid moves with a small reward
            reward += getValidMoveReward();

            // Check if the agent reached the goal
            done = newPosition.equals(getGoalPosition());

            if (done) {
                // If agent reaches the goal, give maximum reward
                reward = 1;
            } else if (!maxStepsReached) {
                // Penalty for stepping into undesired areas (negative cells)
                reward -= (float) (get((int) newPosition.getX(), (int) newPosition.getY()) * 0.6f);
            }
        } else {
            // Penalize invalid moves
            reward -= getInvalidMovePunishment();
        }

        if (maxStepsReached && !done) {
            // Mark episode as done if max steps reached but goal not found
            done = true;
        }

        // Clamp reward to be within -1 to 1 range
        reward = math.clamp(math.scale(reward, minReward, maxReward, -1, 1), -1, 1);
        return new MoveResult(getState(), reward, done);
    }

    public void set(int x, int y, float value) {
        gridMatrix.set(x, y, value);
    }

    public void set(int i, float value) { // simplifies process for creating the environment
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

    public void setAgentPosition(Vector2 position) {
        this.agentPosition = position;
    }

    public void setGoalPosition(Vector2 position) {
        this.goalPosition = position;
    }

    public void setStartPosition(Vector2 position) {
        this.startPosition = position;
    }

    public Vector2 getAgentPosition() {
        return agentPosition;
    }

    public Vector2 getGoalPosition() {
        return goalPosition;
    }

    public Vector2 getStartPosition() {
        return startPosition;
    }

    public boolean isInBounds(int x, int y) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    boolean isValidPositionInBounds(int x, int y) { // by default, any position in bounds is valid
        return isInBounds(x, y);
    }

    public Vector2 getRandomCoordinateInBounds() {
        return new Vector2(math.randomInt(0, width - 1), math.randomInt(0, height - 1));
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
     * @param newPosition The position to be updated. This is a Vector2 object that will be modified
     *                    based on the action.
     */
    static void getNewPosFromAction(int action, Vector2 newPosition) {
        switch(action) {
            case 0: newPosition.addY(-1); break; // Move up    (decrease y)
            case 1: newPosition.addX(1); break;  // Move right (increase x)
            case 2: newPosition.addY(1); break;  // Move down  (increase y)
            case 3: newPosition.addX(-1); break; // Move left  (decrease x)
            case 4: break; // Do nothing
        }
    }

    float getValidMoveReward() {
        return 0.4f;
    }

    float getInvalidMovePunishment() {
        return -0.6f;
    }

    public String toString() {
        return gridMatrix.toString();
    }
}
