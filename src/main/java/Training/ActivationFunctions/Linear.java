package Training.ActivationFunctions;

public class Linear extends ActivationFunction {
    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
