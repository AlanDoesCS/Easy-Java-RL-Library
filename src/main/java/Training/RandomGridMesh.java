package Training;

import java.util.Random;

public class RandomGridMesh extends GridEnvironment {

     public RandomGridMesh(int width, int height) {
         super(width, height);
         fill();
     }

    @Override
    public void fill() {
         float minHeight=0;
         float maxHeight=5;

         Random rand = new Random();

         int n = getNumSquares();
         for (int i = 0; i < n; i++) {
             set(i, rand.nextFloat() * (maxHeight - minHeight) + minHeight);
         }
    }
}
