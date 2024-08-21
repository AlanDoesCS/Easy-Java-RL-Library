package Training.ActivationFunctions;

public class LeakyReLU extends ActivationFunction {
    float gradient;
    public LeakyReLU(float gradient) {
        this.gradient = gradient;
    }

    @Override
    public double activate(double x) {
        return x > 0 ? x : gradient * x;
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : gradient;
    }
}
