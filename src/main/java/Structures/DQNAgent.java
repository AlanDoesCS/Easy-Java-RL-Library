package Structures;

import Training.ActivationFunction;

import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.ArrayList;
import Tools.math;

public class DQNAgent {
    private DQN mainDQN, targetDQN;
    private float epsilon;            // exploration rate for epsilon greedy
    private final float epsilonDecay; // rate of change of epsilon
    private final float epsilonMin;
    private final float gamma;        // discount factor - how much future rewards should be prioritised
    private final int stateSpace;     // number of variables used to describe environment state
    private final int actionSpace;    // number of actions the agent can take in the environment
    private final int targetUpdateFrequency; // how often to update target network
    private int stepCounter;

    public DQNAgent(int actionSpace, List<Layer> layers, float initialEpsilon, float epsilonDecay, float epsilonMin, float gamma, float learningRate) {
        this.epsilon = initialEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.gamma = gamma;
        this.stateSpace = layers.getFirst().getInputSize();
        this.actionSpace = actionSpace;
        this.targetUpdateFrequency = 1000;
        this.stepCounter = 0;

        this.mainDQN = new DQN(stateSpace, layers, learningRate);

        // create a deep copy of mainDQN
        this.targetDQN = new DQN(stateSpace, copyLayers(layers), learningRate);
    }

    private List<Layer> copyLayers(List<Layer> layers) {
        List<Layer> copiedLayers = new ArrayList<>(layers.size());
        for (Layer layer : layers) {
            copiedLayers.add(layer.copy());
        }
        return copiedLayers;
    }

    public int chooseAction(Tensor state) {
        if (math.random() < epsilon) {
            return (int) (Math.random() * actionSpace);
        } else {
            Matrix qValues = (Matrix) mainDQN.getOutput(state);
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
        stepCounter++;

        List<Object> layerOutputs = mainDQN.forwardPass(state);
        Matrix currentQValues = (Matrix) layerOutputs.getLast(); // get predicted q values

        Matrix target = currentQValues.copy();
        Matrix nextQValues = (Matrix) targetDQN.getOutput(nextState);

        // get max Q value for next state
        float maxNextQ = math.max(nextQValues);
        float targetValue = done ? reward : reward + gamma * maxNextQ;
        target.set(0, action, targetValue);

        // update network
        mainDQN.backpropagate(state, target, layerOutputs);

        decayEpsilon();

        if (stepCounter % targetUpdateFrequency == 0) {
            DQN.copyNetworkWeightsAndBiases(mainDQN, targetDQN);
            stepCounter = 0;
        }
    }

    private void decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }

    public float getEpsilon() {
        return epsilon;
    }

    public void saveAgent(String filename) {
        mainDQN.saveNN(filename);
    }

    public void loadAgent(String filename) {
        mainDQN.loadNN(filename);
    }
}
