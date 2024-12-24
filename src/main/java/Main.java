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
            trainer = new DDQNAgentTrainer(Set.of(EmptyGridEnvironment.class));
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
                0.95,                          // gamma
                0.001f,                        // learning rate
                0.9999,                        // learning rate decay
                0.0001f,                       // learning rate minimum
                0.0005                         // tau
        );

        ddqnAgent.dumpDQNInfo();
        ddqnAgent.setVerbose(false);
        trainer.trainAgent(
                ddqnAgent,                     // agent
                600000,                        // num episodes
                3000,                          // save period
                200,                           // visualiser update period
                "plot", "ease", "axis_ticks", "show_path" // varargs
        );
    }
}