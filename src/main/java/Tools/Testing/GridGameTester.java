package Tools.Testing;

import Structures.DDQNAgent;
import Structures.Layer;

import java.util.List;
import java.util.Scanner;

/**
 * A class that extends the DDQNAgent to test actions in a GridEnvironment.
 */
public class GridGameTester extends DDQNAgent {

    public GridGameTester(int actionSpace, List<Layer> layers, float initialEpsilon, float epsilonDecay, float epsilonMin, float gamma, float learningRate, float learningRateDecay, float learningRateMin, float tau) {
        super(actionSpace, layers, initialEpsilon, epsilonDecay, epsilonMin, gamma, learningRate, learningRateDecay, learningRateMin, tau);
    }

    @Override
    public int chooseAction(Object state) {
        Scanner scanner = new Scanner(System.in);
        System.out.printf("Enter action (Must be in range [0, %o]): ", actionSpace-1);
        return scanner.nextInt();
    }

    @Override
    public double train(Object state, int action, double reward, Object nextState, boolean done) {
        return 0;
    }

    @Override
    public double getEpsilon() {
        return 0;
    }

    @Override
    public double getLearningRate() {
        return 0;
    }

    @Override
    public void dumpDQNInfo() {}

    @Override
    public void saveAgent(String filename) {}

    @Override
    public void loadAgent(String filename) {}
}
