package Training.ActivationFunctions;

public class ReLU extends ActivationFunction {
    @Override
    public double activate(double x) {
        return (x<0) ? 0: x; // return 0 if negative, or return the number if positive
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}
