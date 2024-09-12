package Structures;

import java.util.List;
import java.util.ArrayList;
import Tools.math;
import Training.Optimizers.Adam;
import Training.Optimizers.Optimizer;

public class DDQNAgent {
    private Optimizer optimizer;
    private DQN onlineDQN, targetDQN;
    private double epsilon;            // exploration rate for epsilon greedy
    private final double epsilonDecay; // rate of change of epsilon
    private final double epsilonMin;
    private final double learningRateDecay;
    private final double learningRateMin;
    private final double gamma;        // discount factor - how much future rewards should be prioritised
    private final double tau;          // soft update parameter
    private final int stateSpace;     // number of variables used to describe environment state
    protected final int actionSpace;    // number of actions the agent can take in the environment
    private final int targetUpdateFrequency; // how often to update target network
    private int stepCounter;
    private boolean isVerbose = false;
    private int dumpCounter = 0;
    private final int dumpFrequency = 20;

    // Variables to handle periodic exploration bursts
    private final int burstFrequency = 10000; // After every 10k steps, increase exploration
    private final double burstEpsilon = 0.5;  // Epsilon value
    private final int burstDuration = 1000;   // Duration in steps
    private int burstStepCounter = 0;         // Counter

    //Q-value clipping range
    private static final double Q_CLIP_MIN = -50.0;
    private static final double Q_CLIP_MAX = 50.0;
    private static final double REWARD_SCALING = 0.1;

    public DDQNAgent(int actionSpace, List<Layer> layers, double initialEpsilon, double epsilonDecay, double epsilonMin, double gamma, double learningRate, double learningRateDecay, double learningRateMin, double tau) {
        this.optimizer = new Adam();
        this.epsilon = initialEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.learningRateDecay = learningRateDecay;
        this.learningRateMin = learningRateMin;
        this.gamma = gamma;
        this.stateSpace = layers.getFirst().getInputSize();
        this.actionSpace = actionSpace;
        this.targetUpdateFrequency = 5;
        this.tau = tau;
        this.stepCounter = 0;

        this.onlineDQN = new DQN(stateSpace, layers, learningRate);
        this.targetDQN = new DQN(stateSpace, copyLayers(layers), learningRate);
        onlineDQN.setOptimizer(optimizer);
        targetDQN.setOptimizer(optimizer);
    }

    private List<Layer> copyLayers(List<Layer> layers) {
        List<Layer> copiedLayers = new ArrayList<>(layers.size());
        for (Layer layer : layers) {
            copiedLayers.add(layer.copy());
        }
        return copiedLayers;
    }

    @Deprecated
    public void dumpDQNInfo() {
        for (Layer layer : onlineDQN.getLayers()) {
            layer.dumpInfo();
        }
    }

    public int chooseAction(Object state) {
        if (Math.random() < epsilon) {
            return (int) (Math.random() * actionSpace);  // Exploration
        } else {
            MatrixDouble qValues = (MatrixDouble) onlineDQN.getOutput(state);
            return (int) math.maxIndex(qValues).y;  // Exploitation: max Q-value
        }
    }

    private void softUpdate() {
        // Soft update of the target DQN
        for (int i = 0; i < onlineDQN.numLayers(); i++) {
            Layer onlineLayer = onlineDQN.getLayer(i);
            Layer targetLayer = targetDQN.getLayer(i);

            onlineLayer.copyTo(targetLayer, false);  // Copy weights with tau

            if (onlineLayer instanceof MLPLayer) {
                MLPLayer onlineMLP = (MLPLayer) onlineLayer;
                MLPLayer targetMLP = (MLPLayer) targetLayer;

                for (int r = 0; r < onlineMLP.weights.rows; r++) {
                    for (int c = 0; c < onlineMLP.weights.cols; c++) {
                        double onlineWeight = onlineMLP.weights.get(c, r);
                        double targetWeight = targetMLP.weights.get(c, r);
                        targetMLP.weights.set(c, r, tau * onlineWeight + (1 - tau) * targetWeight);
                    }
                }

                for (int r = 0; r < onlineMLP.biases.rows; r++) {
                    double onlineBias = onlineMLP.biases.get(0, r);
                    double targetBias = targetMLP.biases.get(0, r);
                    targetMLP.biases.set(0, r, tau * onlineBias + (1 - tau) * targetBias);
                }
            }
        }
        targetDQN.setLearningRate(onlineDQN.getLearningRate());
    }

    public double train(Object state, int action, double reward, Object nextState, boolean done) {
        stepCounter++;

        List<Object> layerOutputs = onlineDQN.forwardPass(state);
        MatrixDouble currentQValues = (MatrixDouble) layerOutputs.getLast();
        MatrixDouble target = currentQValues.copy();

        if (!done) {
            MatrixDouble nextQValuesOnline = (MatrixDouble) onlineDQN.getOutput(nextState);
            int bestAction = (int) math.maxIndex(nextQValuesOnline).y;
            MatrixDouble nextQValuesTarget = (MatrixDouble) targetDQN.getOutput(nextState);
            double targetQ = nextQValuesTarget.get(0, bestAction);
            double targetValue = reward + gamma * targetQ;
            target.set(0, action, targetValue);
        } else {
            target.set(0, action, reward);
        }

        // Update epsilon after training step
        decayEpsilon();
        decayLearningRate();

        // Soft update for target network
        if (stepCounter % targetUpdateFrequency == 0) {
            softUpdate();
        }
        return target.get(0, action) - currentQValues.get(0, action);
    }

    private void decayEpsilon() {
        // Introduce periodic exploration bursts
        if (stepCounter % burstFrequency == 0) {
            epsilon = burstEpsilon;  // Re-increase epsilon for exploration burst
            burstStepCounter = 0;
        }

        if (burstStepCounter < burstDuration) {
            burstStepCounter++;
        } else {
            epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);  // Gradually decay epsilon back to its minimum
        }
    }

    private void decayLearningRate() {
        onlineDQN.setLearningRate(Math.max(learningRateMin, onlineDQN.getLearningRate() * learningRateDecay));
    }

    public double getEpsilon() {
        return epsilon;
    }
    public double getLearningRate() { return onlineDQN.getLearningRate(); }
    public void setVerbose(boolean verbose) {
        isVerbose = verbose;
    }

    public void saveAgent(String filename) {
        onlineDQN.saveNN(filename);
    }

    public void loadAgent(String filename) {
        onlineDQN.loadNN(filename);
    }

    public DQN getOnlineDQN() {
        return onlineDQN;
    }
    public DQN getTargetDQN() {
        return targetDQN;
    }

    public double getMinEpsilon() {
        return epsilonMin;
    }

    public void setEpsilon(double newEpsilon) {
        this.epsilon = Math.max(epsilonMin, Math.min(1.0, newEpsilon));
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }
}
