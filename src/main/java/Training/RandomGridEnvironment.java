package Training;

import Structures.Vector2D;
import Tools.math;

import java.util.Random;

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
