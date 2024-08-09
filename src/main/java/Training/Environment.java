package Training;

import Structures.Matrix;
import com.sun.jdi.InvalidTypeException;
import jdk.jshell.spi.ExecutionControl;

public abstract class Environment {
    // defaults
    static int gridWidth = 30, gridHeight = 30;
    static int octaves = 8;
    static int stateSpace=-1, actionSpace=-1;
    static float persistence = 0.9f, step = 0.01f;

    public static int getActionSpace() {
        return actionSpace;
    }
    public static int getStateSpace() {
        return stateSpace;
    }
    public static void setActionSpace(int actionSpace) {
        if (actionSpace < 1) throw new IllegalArgumentException("actionSpace must be greater than 1, not "+actionSpace);
        Environment.actionSpace = actionSpace;
    }
    public static void setStateSpace(int stateSpace) {
        if (stateSpace < 1) throw new IllegalArgumentException("stateSpace must be greater than 1, not "+stateSpace);
        Environment.stateSpace = stateSpace;
    }

    public static int getGridSquares() {
        return gridWidth*gridHeight;
    }

    public static int getGridWidth() {
        return gridWidth;
    }
    public static int getGridHeight() {
        return gridHeight;
    }

    /*
        Randomize:
        Randomize the environment

        - MUST ALSO RANDOMIZE START AND TARGET POSITIONS
    */
    public abstract void randomize();
    public abstract Matrix getState();

    // Environment instantiation table
    // Todo: should probably make a bit more expandable
    public static Environment of(Class<? extends Environment> envClass) throws InvalidTypeException {
        if (envClass.equals(MazeGridEnvironment.class)) {
            return new MazeGridEnvironment(gridWidth, gridHeight);
        } else if (envClass.equals(RandomGridEnvironment.class)) {
            return new RandomGridEnvironment(gridWidth, gridHeight);
        } else if (envClass.equals(PerlinGridEnvironment.class)) {
            return new PerlinGridEnvironment(gridWidth, gridHeight, octaves, persistence, step);
        } else {
            throw new InvalidTypeException("Not a recognised Environment class");
        }
    }

    /*
    -------------------------------------------------------

    ACCESSORS AND MUTATORS

    -------------------------------------------------------
    */

    public static void setGridWidth(int width) {
        gridWidth = width;
    }

    public static void setGridHeight(int height) {
        gridHeight = height;
    }

    public static void setDimensions(int width, int height) {
        gridWidth = width;
        gridHeight = height;
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

    public abstract MoveResult step(int action);

    public String getType() {
        return this.getClass().getSimpleName();
    }

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
}
