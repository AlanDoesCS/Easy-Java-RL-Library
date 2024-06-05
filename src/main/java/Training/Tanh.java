package Training;

public class Tanh extends ActivationFunction {
    @Override
    public float activate(float x) {
        return (float) Math.tanh(x);
    }

    @Override
    public float derivative(float x) {
        float tanh = (float) Math.tanh(x);
        return 1 - (tanh * tanh);
    }
}
