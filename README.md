# Easy Java Reinforcement Learning Library (EJRLL)
A simple neural network library for training Deep Q Networks in different environments.

![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/AlanDoesCS/RL-Pathfinding)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/AlanDoesCS/RL-Pathfinding)
![GitHub contributors](https://img.shields.io/github/contributors/AlanDoesCS/RL-Pathfinding)
![GitHub Repo stars](https://img.shields.io/github/stars/AlanDoesCS/RL-Pathfinding)
![GitHub forks](https://img.shields.io/github/forks/AlanDoesCS/RL-Pathfinding)

---

## Environments
- 2D perlin noise
- Maze (generated using recursive backtracking)
- Pseudorandom noise

---

## RL Algorithms
- DQN
- Double DQN

## Replay
- Replay Buffer
- Prioritized Experience Replay

## Optimizers
- Adam

---

## Example usage:

```java
public class Main {
    public static void main(String[] args) {
        Environment.setStateType(Environment.StateType.PositionVectorOnly);
        Environment.setDimensions(10, 10);
        Environment.setActionSpace(4);

        DDQNAgentTrainer trainer;
        try {
            trainer = new DDQNAgentTrainer(Set.of(EmptyGridEnvironment.class, RandomGridEnvironment.class, PerlinGridEnvironment.class, MazeGridEnvironment.class));
        } catch (InvalidTypeException e) {
            e.printStackTrace();
            return;
        }

        List<Layer> layers = new ArrayList<>();
        LeakyReLU leakyRelu = new LeakyReLU(0.1f);
        float lambda = 0.0001f;

        // StateSpace is 104, ActionSpace is 5
        layers.add(new MLPLayer(Environment.getStateSpace(), 64, leakyRelu, 0, lambda));
        layers.add(new MLPLayer(64, 64, leakyRelu, 0, lambda));
        layers.add(new MLPLayer(64, Environment.getActionSpace(), new Linear(), 0, lambda));

        DDQNAgent ddqnAgent = new DDQNAgent(
                Environment.getActionSpace(),  // action space
                layers,                        // layers
                1,                             // initial epsilon
                0.9999,                        // epsilon decay
                0.01,                          // epsilon min
                0.999,                         // gamma
                0.0001,                        // learning rate
                0.99995,                       // learning rate decay
                0.000001f,                     // learning rate minimum
                0.005                          // tau
        );

        trainer.trainAgent(
                ddqnAgent,                     // agent
                600000,                        // num episodes
                500,                           // save period
                1,                             // visualiser update period
                "plot", "ease", "axis_ticks", "show_path", "verbose" // varargs
        );
    }
}
```

## Papers & Resources Used
This list is incomplete, but I will try and ensure I add all the sources I used eventually

### Papers
- https://arxiv.org/pdf/1511.08458
- http://arxiv.org/pdf/1511.05952v4
- https://arxiv.org/pdf/1412.6980
- http://arxiv.org/pdf/1502.03167
- https://proceedings.neurips.cc/paper/2020/file/32fcc8cfe1fa4c77b5c58dafd36d1a98-Paper.pdf
- https://doi.org/10.1109/ACCESS.2019.2941229
- http://www.arxiv.org/pdf/1509.06461
- https://arxiv.org/pdf/1312.5602

### Videos
- https://www.youtube.com/watch?v=z9hJzduHToc
- https://www.youtube.com/watch?v=s2coXdufOzE
- https://youtu.be/ECV5yeigZIg?si=3EXfuIGTH2BABkeS
