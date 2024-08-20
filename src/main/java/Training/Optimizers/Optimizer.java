package Training.Optimizers;

import Structures.Layer;

public abstract class Optimizer {
    public abstract void optimize(Layer layer, float alpha);
}
