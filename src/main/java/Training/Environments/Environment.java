package Training.Environments;

import com.sun.jdi.InvalidTypeException;

/**
 * Environment
 * <p>
 * Abstract class for defining the environment in which the agent will operate.
 * Only allows for one state and action space to be defined.
 */
public abstract class Environment {
    // Static StateType enum that defines how the environment's state is represented
    public static enum StateType {
        PositionVectorOnly,    // Only the agent and goal positions as a vector
        PositionAndGridAsColumn,  // Positions with the entire grid as a column matrix
        PositionAndGridAsLayers   // The grid represented as multiple layers for agent, goal, and environment
    }

    static StateType stateType = StateType.PositionVectorOnly;  // Default state type

    // Environment variables
    static int gridWidth = 30, gridHeight = 30;  // Grid dimensions
    static int octaves = 8;  // Octaves for Perlin noise environments
    static int stateSpace = -1, actionSpace = -1;  // State and action space dimensions
    static float persistence = 0.9f, step = 0.01f;  // Perlin noise parameters
    float minReward, maxReward;  // Reward scaling

    // Sets the type of state representation (column matrix, position vector, etc.)
    public static void setStateType(StateType stateType) {
        Environment.stateType = stateType;
        updateStateSpace();  // Automatically update state space when state type changes
    }

    // Updates the state space dimension based on the selected StateType
    private static void updateStateSpace() {
        switch (stateType) {
            case PositionVectorOnly:
                stateSpace = 4;  // Just the agent and goal positions
                break;
            case PositionAndGridAsColumn:
                stateSpace = 4 + getGridSquares();  // Agent, goal, and the entire grid
                break;
            case PositionAndGridAsLayers:
                stateSpace = 3 * getGridWidth() * getGridHeight();  // Environment, agent, and goal as layers
                break;
        }
    }

    // Getters and setters for various grid and environment parameters

    public static int getActionSpace() {
        return actionSpace;
    }

    public static int getStateSpace() {
        return stateSpace;
    }

    public static void setActionSpace(int actionSpace) {
        if (actionSpace < 1) throw new IllegalArgumentException("actionSpace must be greater than 1, not " + actionSpace);
        Environment.actionSpace = actionSpace;
    }

    public static void setStateSpace(int stateSpace) {
        if (stateSpace < 1) throw new IllegalArgumentException("stateSpace must be greater than 1, not " + stateSpace);
        Environment.stateSpace = stateSpace;
    }

    public static int getGridSquares() {
        return gridWidth * gridHeight;
    }

    public static int getGridWidth() {
        return gridWidth;
    }

    public static int getGridHeight() {
        return gridHeight;
    }

    public static void setGridWidth(int width) {
        gridWidth = width;
    }

    public static void setGridHeight(int height) {
        gridHeight = height;
    }

    public static void setDimensions(int width, int height) {
        gridWidth = width;
        gridHeight = height;
        updateStateSpace();  // Recalculate state space when dimensions change
    }

    public static void setOctaves(int octaves) {
        Environment.octaves = octaves;
    }

    public static void setPersistence(float persistence) {
        Environment.persistence = persistence;
    }

    public static void setStep(float step) {
        Environment.step = step;
    }

    /*
     * Randomize:
     * Randomize the environment and agent, including random start and goal positions.
     */
    public abstract void randomize();

    /*
     * Environment instantiation:
     * Create a specific environment based on the class type.
     */
    public static Environment of(Class<? extends Environment> envClass) throws InvalidTypeException {
        if (envClass.equals(MazeGridEnvironment.class)) {
            return new MazeGridEnvironment(gridWidth, gridHeight);
        } else if (envClass.equals(RandomGridEnvironment.class)) {
            return new RandomGridEnvironment(gridWidth, gridHeight);
        } else if (envClass.equals(EmptyGridEnvironment.class)) {
            return new EmptyGridEnvironment(gridWidth, gridHeight);
        } else if (envClass.equals(PerlinGridEnvironment.class)) {
            return new PerlinGridEnvironment(gridWidth, gridHeight, octaves, persistence, step);
        } else {
            throw new InvalidTypeException("Not a recognized Environment class");
        }
    }

    /*
     * MoveResult:
     * Defines the result of an agent's move, including the next state, reward, and whether the episode is done.
     */
    public static class MoveResult {
        public Object state;  // Next state after the move
        public float reward;  // Reward received after the move
        public boolean done;  // Whether the episode is finished (e.g., goal reached or max steps)

        public MoveResult(Object state, float reward, boolean done) {
            this.state = state;
            this.reward = reward;
            this.done = done;
        }
    }

    /*
     * Abstract step method:
     * Each environment subclass must define this to handle the agent's step, update its position, calculate rewards, etc.
     */
    public abstract MoveResult step(int action);

    public String getType() {
        return this.getClass().getSimpleName();
    }
}
