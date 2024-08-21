package Structures;

import java.util.List;
import java.util.ArrayList;
import Tools.math;
import Training.Optimizers.Adam;
import Training.Optimizers.Optimizer;

public class DDQNAgent {
    private Optimizer optimizer;
    private DQN onlineDQN, targetDQN;
    private float epsilon;            // exploration rate for epsilon greedy
    private final float epsilonDecay; // rate of change of epsilon
    private final float epsilonMin;
    private final float learningRateDecay;
    private final float learningRateMin;
    private final float gamma;        // discount factor - how much future rewards should be prioritised
    private final float tau;          // soft update parameter
    private final int stateSpace;     // number of variables used to describe environment state
    protected final int actionSpace;    // number of actions the agent can take in the environment
    private final int targetUpdateFrequency; // how often to update target network
    private int stepCounter;

    private int dumpCounter = 0;
    private final int dumpFrequency = 20;

    public DDQNAgent(int actionSpace, List<Layer> layers, float initialEpsilon, float epsilonDecay, float epsilonMin, float gamma, float learningRate, float learningRateDecay, float learningRateMin, float tau) {
        this.optimizer = new Adam();
        this.epsilon = initialEpsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.learningRateDecay = learningRateDecay;
        this.learningRateMin = learningRateMin;
        this.gamma = gamma;
        this.stateSpace = layers.getFirst().getInputSize();
        this.actionSpace = actionSpace;
        this.targetUpdateFrequency = 5000;
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
        if (math.random() < epsilon) {
            return (int) (Math.random() * actionSpace);
        } else {
            Matrix qValues = (Matrix) onlineDQN.getOutput(state);
            if (dumpCounter % dumpFrequency == 0) {
                System.out.println("Q Values: "+qValues.toRowMatrix() + ", maxIndex = "+math.maxIndex(qValues).y);
            }
            dumpCounter++;

            return (int) math.maxIndex(qValues).y;
        }
    }

    private void softUpdate() {
        for (int i = 0; i < onlineDQN.numLayers(); i++) {
            Layer onlineLayer = onlineDQN.getLayer(i);
            Layer targetLayer = targetDQN.getLayer(i);

            onlineLayer.copyTo(targetLayer, false);

            if (onlineLayer instanceof MLPLayer) {
                MLPLayer onlineMLP = (MLPLayer) onlineLayer;
                MLPLayer targetMLP = (MLPLayer) targetLayer;

                for (int r = 0; r < onlineMLP.weights.rows; r++) {
                    for (int c = 0; c < onlineMLP.weights.cols; c++) {
                        float onlineWeight = onlineMLP.weights.get(c, r);
                        float targetWeight = targetMLP.weights.get(c, r);
                        targetMLP.weights.set(c, r, tau * onlineWeight + (1 - tau) * targetWeight);
                    }
                }

                for (int r = 0; r < onlineMLP.biases.rows; r++) {
                    float onlineBias = onlineMLP.biases.get(0, r);
                    float targetBias = targetMLP.biases.get(0, r);
                    targetMLP.biases.set(0, r, tau * onlineBias + (1 - tau) * targetBias);
                }
            } else if (onlineLayer instanceof ConvLayer) {
                ConvLayer onlineConv = (ConvLayer) onlineLayer;
                ConvLayer targetConv = (ConvLayer) targetLayer;

                for (int f = 0; f < onlineConv.filters.length; f++) {
                    for (int d = 0; d < onlineConv.filters[f].length; d++) {
                        for (int h = 0; h < onlineConv.filters[f][d].length; h++) {
                            for (int w = 0; w < onlineConv.filters[f][d][h].length; w++) {
                                float onlineFilter = onlineConv.filters[f][d][h][w];
                                float targetFilter = targetConv.filters[f][d][h][w];
                                targetConv.filters[f][d][h][w] = tau * onlineFilter + (1 - tau) * targetFilter;
                            }
                        }
                    }
                }

                for (int f = 0; f < onlineConv.biases.length; f++) {
                    float onlineBias = onlineConv.biases[f];
                    float targetBias = targetConv.biases[f];
                    targetConv.biases[f] = tau * onlineBias + (1 - tau) * targetBias;
                }
            }
        }
        targetDQN.setLearningRate(onlineDQN.getLearningRate());
    }

    public float train(Object state, int action, float reward, Object nextState, boolean done) {
        stepCounter++;

        List<Object> layerOutputs = onlineDQN.forwardPass(state);
        Matrix currentQValues = (Matrix) layerOutputs.getLast(); // get predicted q values
        Matrix target = currentQValues.copy();

        if (!done) {
            // use online to select best action
            Matrix nextQValuesOnline = (Matrix) onlineDQN.getOutput(nextState);
            int bestAction = (int) math.maxIndex(nextQValuesOnline).y;

            // use target to evaluate Q val of best action
            Matrix nextQValuesTarget = (Matrix) targetDQN.getOutput(nextState);
            float targetQ = nextQValuesTarget.get(0, bestAction);

            // calculate target value
            float targetValue = reward + gamma * targetQ;
            target.set(0, action, targetValue);
        } else {
            target.set(0, action, reward);
        }

        // Calculate TD error
        float tdError = target.get(0, action) - currentQValues.get(0, action);
        if (Float.isNaN(currentQValues.get(0, action))) {
            System.err.println("action " + action + " is NaN!! : " + currentQValues.get(0, action));
        }

        Object gradientOutput = Matrix.subtract(target, currentQValues);

        for (int i = onlineDQN.numLayers() - 1; i >= 0; i--) {
            Layer layer = onlineDQN.getLayer(i);
            Object layerInput = layerOutputs.get(i);

            gradientOutput = layer.backpropagate(layerInput, gradientOutput);
            optimizer.optimize(layer);
        }

        decayEpsilon();
        decayLearningRate();

        if (stepCounter % targetUpdateFrequency == 0) {
            softUpdate();
            stepCounter = 0;
        }

        return tdError;
    }

    private void decayEpsilon() {
        epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
    }
    private void decayLearningRate() {
        onlineDQN.setLearningRate(Math.max(learningRateMin, onlineDQN.getLearningRate() * learningRateDecay));
    }

    public float getEpsilon() {
        return epsilon;
    }
    public float getLearningRate() { return onlineDQN.getLearningRate(); }

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
}
