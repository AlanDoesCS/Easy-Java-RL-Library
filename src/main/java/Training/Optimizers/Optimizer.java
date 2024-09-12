package Training.Optimizers;

import Structures.Layer;

public abstract class Optimizer {
    protected int t = 0;
    public void incrementT() { t++; }

    public abstract void optimize(Layer layer);
}
