package Training;

public class Linear extends ActivationFunction {
    @Override
    public float activate(float x) {
        return x;
    }

    @Override
    public float derivative(float x) {
        return 1;
    }
}
