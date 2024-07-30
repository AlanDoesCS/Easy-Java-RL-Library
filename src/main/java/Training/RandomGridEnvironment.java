package Training;

import Structures.Vector2D;

import java.util.Random;

public class RandomGridEnvironment extends GridEnvironment {

     public RandomGridEnvironment(int width, int height) {
         super(width, height);
         fill();
     }

     @Override
     public void fill() {
         Random rand = new Random();

         int n = getNumSquares();
         for (int i = 0; i < n; i++) {
             set(i, rand.nextFloat());
         }
     }

    @Override
    public MoveResult step(int action) {
        Vector2D currentPosition = getAgentPosition();
        Vector2D newPosition = currentPosition.copy();
        getNewPosFromAction(action, newPosition);

        if (isInBounds((int)newPosition.getI(), (int)newPosition.getJ())) {
            setAgentPosition(newPosition);
        }

        boolean done = newPosition.equals(getGoalPosition());

        float reward = done ? getCompletionReward() : get((int)newPosition.getI(), (int)newPosition.getJ());

        return new MoveResult(getState(), reward, done);
    }
}
