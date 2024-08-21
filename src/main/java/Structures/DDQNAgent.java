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

    private int dumpCounter = 0;
    private final int dumpFrequency = 20;

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
            MatrixDouble qValues = (MatrixDouble) onlineDQN.getOutput(state);
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
            } else if (onlineLayer instanceof ConvLayer) {
                ConvLayer onlineConv = (ConvLayer) onlineLayer;
                ConvLayer targetConv = (ConvLayer) targetLayer;

                for (int f = 0; f < onlineConv.filters.length; f++) {
                    for (int d = 0; d < onlineConv.filters[f].length; d++) {
                        for (int h = 0; h < onlineConv.filters[f][d].length; h++) {
                            for (int w = 0; w < onlineConv.filters[f][d][h].length; w++) {
                                double onlineFilter = onlineConv.filters[f][d][h][w];
                                double targetFilter = targetConv.filters[f][d][h][w];
                                targetConv.filters[f][d][h][w] = tau * onlineFilter + (1 - tau) * targetFilter;
                            }
                        }
                    }
                }

                for (int f = 0; f < onlineConv.biases.length; f++) {
                    double onlineBias = onlineConv.biases[f];
                    double targetBias = targetConv.biases[f];
                    targetConv.biases[f] = tau * onlineBias + (1 - tau) * targetBias;
                }
            }
        }
        targetDQN.setLearningRate(onlineDQN.getLearningRate());
    }

    public double train(Object state, int action, double reward, Object nextState, boolean done) {
        stepCounter++;

        List<Object> layerOutputs = onlineDQN.forwardPass(state);
        MatrixDouble currentQValues = (MatrixDouble) layerOutputs.getLast(); // get predicted q values
        MatrixDouble target = currentQValues.copy();

        if (!done) {
            // use online to select best action
            MatrixDouble nextQValuesOnline = (MatrixDouble) onlineDQN.getOutput(nextState);
            int bestAction = (int) math.maxIndex(nextQValuesOnline).y;

            // use target to evaluate Q val of best action
            MatrixDouble nextQValuesTarget = (MatrixDouble) targetDQN.getOutput(nextState);
            double targetQ = nextQValuesTarget.get(0, bestAction);

            // calculate target value
            double targetValue = reward + gamma * targetQ;
            target.set(0, action, targetValue);
        } else {
            target.set(0, action, reward);
        }

        // Calculate TD error
        double tdError = target.get(0, action) - currentQValues.get(0, action);
        if (Double.isNaN(currentQValues.get(0, action))) {
            System.err.println("action " + action + " is NaN!! : " + currentQValues.get(0, action));
        }

        Object gradientOutput = MatrixDouble.subtract(target, currentQValues);

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

    public double getEpsilon() {
        return epsilon;
    }
    public double getLearningRate() { return onlineDQN.getLearningRate(); }

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
