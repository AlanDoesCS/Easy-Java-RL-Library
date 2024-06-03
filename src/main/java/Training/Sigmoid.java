package Training;

public class Sigmoid extends ActivationFunction {

    @Override
    public float activate(float x) {
        return (float) (1f / (1 + Math.exp(-x)));
    }

    @Override
    public float derivative(float x) {
        float sigma = activate(x);
        return sigma * (1 - sigma);
    }
}
