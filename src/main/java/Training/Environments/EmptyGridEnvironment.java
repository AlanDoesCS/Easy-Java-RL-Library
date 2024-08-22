package Training.Environments;

public class EmptyGridEnvironment extends GridEnvironment {

     public EmptyGridEnvironment(int width, int height) {
         super(width, height);
         fill();
     }

     @Override
     public void fill() {
         int n = getNumSquares();
         for (int i = 0; i < n; i++) {
             set(i, 0);
         }
     }
}
