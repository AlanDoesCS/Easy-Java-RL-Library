package Training;

public class ReLU extends ActivationFunction {
    @Override
    public float activate(float x) {
        return (x<0) ? 0: x; // return 0 if negative, or return the number if positive
    }

    @Override
    public float derivative(float x) {
        return 1;
    }

}
