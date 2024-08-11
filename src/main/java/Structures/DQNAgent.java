package Structures;

import Training.ActivationFunction;

import java.util.List;
import Tools.math;

public class DQNAgent {
    private DQN dqn;
    private final float epsilon;      // exploration rate for epsilon greedy
    private final float gamma;        // discount factor - how much future rewards should be prioritised
    private final int stateSpace;     // number of variables used to describe environment state
    private final int actionSpace;    // number of actions the agent can take in the environment

    public DQNAgent(int actionSpace, List<Layer> layers, float epsilon, float gamma, float learningRate) {
        this.epsilon = epsilon;
        this.gamma = gamma;
        this.stateSpace = layers.getFirst().getInputSize();
        this.actionSpace = actionSpace;

        this.dqn = new DQN(stateSpace, layers, learningRate);
    }

    public int chooseAction(Tensor state) {
        if (math.random() < epsilon) {
            return (int) (Math.random() * actionSpace);
        } else {
            Matrix qValues = (Matrix) dqn.getOutput(state);
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

    public void train(Object state, int action, float reward, Object nextState, boolean done) {
        List<Object> layerOutputs = dqn.forwardPass(state);
        Matrix currentQValues = (Matrix) layerOutputs.getLast(); // get predicted q values

        Matrix target = currentQValues.copy();
        Matrix nextQValues = (Matrix) dqn.getOutput(nextState);

        // get max Q value for next state
        float maxNextQ = nextQValues.get(0, 0);
        for (int i = 1; i < actionSpace; i++) {
            if (nextQValues.get(0, i) > maxNextQ) {
                maxNextQ = nextQValues.get(0, i);
            }
        }

        float targetValue = done ? reward : reward + gamma * maxNextQ;
        target.set(0, action, targetValue);

        // update network
        dqn.backpropagate(state, target, layerOutputs);
    }

    public void saveAgent(String filename) {
        dqn.saveNN(filename);
    }

    public void loadAgent(String filename) {
        dqn.loadNN(filename);
    }
}
