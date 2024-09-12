package Training.Environments;

import Tools.math;

public class RandomGridEnvironment extends GridEnvironment {

    public RandomGridEnvironment(int width, int height) {
        super(width, height);
        fill();
    }

    @Override
    public void fill() {
        int n = getNumSquares();
        for (int i = 0; i < n; i++) {
            set(i, math.randomFloat(0, 1));
        }
    }
}
