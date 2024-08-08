package Training;

import Structures.Matrix;
import Structures.Vector2D;
import Tools.Perlin2D;
import Tools.math;

public class PerlinGridEnvironment extends GridEnvironment {
    float step, persistence;
    int octaves;
    Perlin2D perlin;
    Matrix gridAsMatrix;

    public PerlinGridEnvironment(int width, int height, int octaves, float persistence, float step) {
        super(width, height);

        this.perlin = new Perlin2D(octaves, persistence);
        this.step = step;
        this.octaves = octaves;
        this.persistence = persistence;

        fill();
    }

    @Override
    void fill() {
        gridAsMatrix = fill(new Matrix(height, width));
    }

    Matrix fill(Matrix destination) {
        if (destination == null) throw new NullPointerException("Matrix destination is null!");

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                float noiseValue = perlin.noise(x*step, y*step, 0, 1);
                set(x, y, noiseValue);
                destination.set(x, y, noiseValue);
            }
        }
        return destination;
    }

    @Override
    public void refill() {
        this.perlin = new Perlin2D(octaves, persistence);
        fill();
    }

    @Override
    public MoveResult step(int action) {
        Vector2D currentPosition = getAgentPosition();
        Vector2D newPosition = currentPosition.copy();
        getNewPosFromAction(action, newPosition);

        if (isInBounds((int)newPosition.getI(), (int)newPosition.getJ())) {
            setAgentPosition(newPosition);
        } else {
            newPosition = currentPosition;
        }

        boolean done = newPosition.equals(getGoalPosition());

        float reward = done ? getCompletionReward() : get((int)newPosition.getI(), (int)newPosition.getJ());

        return new MoveResult(getState(), reward, done);
    }
}
