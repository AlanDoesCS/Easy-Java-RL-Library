package Training;

import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {
    abstract public float activate(float x);
    abstract public float derivative(float x);
}
