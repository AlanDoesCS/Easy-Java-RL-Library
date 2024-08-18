package Training;

public class LeakyReLU extends ActivationFunction {
    float gradient;
    public LeakyReLU(float gradient) {
        this.gradient = gradient;
    }

    @Override
    public float activate(float x) {
        return x > 0 ? x : gradient * x;
    }

    @Override
    public float derivative(float x) {
        return x > 0 ? 1 : gradient;
    }
}
