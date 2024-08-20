# RL-Pathfinding
A simple neural network library for training networks in different environments.

---

## Environments
- 2D perlin noise
- Maze (generated using recursive backtracking)
- Psudorandom noise

---

## Example usage:

```java
import Structures.*;
import Tools.Environment_Visualiser;
import Tools.Pathfinding.Pathfinder;
import Training.*;

import com.sun.jdi.InvalidTypeException;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        Environment.setDimensions(10, 10);
        Environment.setStateSpace(Environment.getGridSquares() + 4);
        Environment.setActionSpace(4);

        Trainer trainer;
        try {
            trainer = new Trainer(Set.of(PerlinGridEnvironment.class, MazeGridEnvironment.class, RandomGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        ActivationFunction sig = new Sigmoid();
        ActivationFunction relu = new ReLU();

        List<Layer> layers = Layer.createHiddenLayers(
                List.of(100, 200, 300),
                List.of(sig, sig, sig),
                List.of(0, 0, 0)
        );

        DDQNAgent DDQNAgent = new DDQNAgent(Environment.getActionSpace(), layers, 0.1f, 0.99f, 0.001f, relu, 0);

        trainer.trainAgent(DDQNAgent, 6000, 1000);
    }

    public static void testPathfinding(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> shortestPath = Pathfinder.dijkstra(start, end, environment);
        Environment_Visualiser vis = new Environment_Visualiser(environment);
        vis.addPath(shortestPath, new Color(0, 128, 128));
    }
}
```
