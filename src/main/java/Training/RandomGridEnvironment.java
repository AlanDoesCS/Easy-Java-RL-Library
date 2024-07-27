package Training;

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
}
