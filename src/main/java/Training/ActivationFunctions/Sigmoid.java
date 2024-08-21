package Training.ActivationFunctions;

public class Sigmoid extends ActivationFunction {

    @Override
    public double activate(double x) {
        return (1d / (1 + Math.exp(-x)));
    }

    @Override
    public double derivative(double x) {
        double sigma = activate(x);
        return sigma * (1 - sigma);
    }
}
