import Structures.*;

import Training.DDQNAgentTrainer;
import Training.ActivationFunctions.*;
import Training.Environments.*;

import com.sun.jdi.InvalidTypeException;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

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