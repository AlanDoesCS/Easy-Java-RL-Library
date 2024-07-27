package Structures;

import Training.ActivationFunction;

import java.util.List;
import Tools.math;

public class DQNAgent {
    private DQN dqn;
    private final float epsilon;      // exploration rate for epsilon greedy
    private final float gamma;        // discount factor
    private final int stateSpace;     // number of variables used to describe environment state
    private final int actionSpace;    // number of actions the agent can take in the environment

    public DQNAgent(int stateSpace, int actionSpace, List<Layer> hiddenLayers, float epsilon, float gamma, float learningRate, ActivationFunction outputActivation, float outputBias) {
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.stateSpace = stateSpace;
        this.actionSpace = actionSpace;

        this.dqn = new DQN(stateSpace, hiddenLayers, actionSpace, learningRate, outputActivation, outputBias);
    }

    public int chooseAction(Matrix state) {
        if (math.random() < epsilon) {
            return (int) (Math.random() * actionSpace);
        } else {
            Matrix qValues = dqn.getOutput(state);
            int actionIndex = 0;
            float max = qValues.get(0, 0);
            for (int i = 1; i < actionSpace; i++) {
                if (qValues.get(0, i) > max) {
                    max = qValues.get(0, i);
                    actionIndex = i;
                }
            }
            return actionIndex;
        }
    }
}
