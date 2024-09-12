package Training;

import Structures.DDQNAgent;
import Structures.MatrixDouble;
import Structures.Vector2;
import Tools.Environment_Visualiser;
import Tools.GraphPlotter;
import Tools.Pathfinding.Pathfinder;
import Tools.math;
import Training.Environments.Environment;
import Training.Environments.GridEnvironment;
import Training.Replay.ExperienceReplay;
import Training.Replay.PrioritizedExperienceReplay;
import com.sun.jdi.InvalidTypeException;
import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import javax.swing.*;

public class DDQNAgentTrainer {
    Set<Class<? extends GridEnvironment>> environmentClasses;

    public DDQNAgentTrainer(Set<Class<? extends GridEnvironment>> environments) throws InvalidTypeException {
        this.environmentClasses = environments;
    }

    /**
     * Trains the DQN agent using the specified parameters.
     *
     * @param agent                  the DQN agent to be trained
     * @param numEpisodes            the number of episodes to train the agent
     * @param savePeriod             the period (in episodes) at which the agent's state is saved
     * @param visualiserUpdatePeriod the period (in episodes) at which the visualiser is updated
     * @param varargs                additional arguments for training options (e.g., "verbose", "plot", "show_path")
     */
    public void trainAgent(DDQNAgent agent, int numEpisodes, int savePeriod, int visualiserUpdatePeriod, String... varargs) {
        List<String> args = Arrays.asList(varargs);
        boolean isVerbose = args.contains("verbose");  // Declared verbose flag here
        boolean dumpInfo = args.contains("dump_info");

        boolean plot = args.contains("plot");
        GraphPlotter averageRewardPlotter = null;
        GraphPlotter averageLossPlotter = null;

        if (plot) {
            averageRewardPlotter = new GraphPlotter("Average Reward vs Episodes", GraphPlotter.Types.LINE, "Episode", "Average Reward", varargs);
            averageRewardPlotter.setVisible(true);
            averageRewardPlotter.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

            averageLossPlotter = new GraphPlotter("Average Loss vs Episodes", GraphPlotter.Types.LINE, "Episode", "Average Loss", varargs);
            averageLossPlotter.setVisible(true);
            averageLossPlotter.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        }

        boolean showPath = args.contains("show_path");
        Environment_Visualiser visualiser = null;  // for showing the path - Dijkstra's path
        Environment_Visualiser visualiser2 = null; // for showing the path - DQN's path

        if (isVerbose) {
            System.out.println("Training agent with "+numEpisodes+" episodes, saving every "+savePeriod+" episodes.");
            System.out.println("plot: "+plot+", show_path: "+showPath+"\n");
        }

        List<GridEnvironment> environments = new ArrayList<>();

        try {
            for (Class<? extends GridEnvironment> envClass : environmentClasses) {
                environments.add((GridEnvironment) Environment.of(envClass));
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // TRAINING LOOP -----------------------------------------------------------------------------------------------

        PrioritizedExperienceReplay replay = new PrioritizedExperienceReplay(200000);
        int batchSize = 32;

        for (int episode = 1; episode <= numEpisodes; episode++) {
            double totalSquaredTDError = 0;
            int tdErrorCounter = 0;

            GridEnvironment environment = environments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            MatrixDouble state = (MatrixDouble) environment.getState();
            boolean done = false;
            double cumulativeReward = 0;
            ArrayList<Vector2> dqnPath = new ArrayList<>();

            while (!done) {
                int action = agent.chooseAction(state);
                if (!dqnPath.isEmpty()) {
                    if (!dqnPath.get(dqnPath.size() - 1).equals(environment.getAgentPosition())) {
                        dqnPath.add(environment.getAgentPosition()); // Add to path if agent has moved
                    }
                } else {
                    dqnPath.add(environment.getAgentPosition());
                }

                Environment.MoveResult result = environment.step(action);

                // Add experience to replay buffer
                replay.add(new ExperienceReplay.Experience(state, action, result.reward, result.state, result.done));

                if (replay.size() > batchSize) {
                    List<ExperienceReplay.Experience> batch = replay.sample(batchSize);
                    List<Integer> treeIndices = new ArrayList<>();
                    List<Double> tdErrors = new ArrayList<>();

                    for (ExperienceReplay.Experience exp : batch) {
                        double tdError = agent.train(exp.state, exp.action, exp.reward, exp.nextState, exp.done);
                        treeIndices.add(exp.index);
                        tdErrors.add(tdError);

                        totalSquaredTDError += tdError * tdError;
                        tdErrorCounter++;
                    }

                    replay.updatePriorities(treeIndices, tdErrors);
                }

                // Update state and cumulative reward
                state = (MatrixDouble) result.state;
                done = result.done;
                cumulativeReward += result.reward;

                MatrixDouble qValues = (MatrixDouble) agent.getOnlineDQN().getOutput(state);


                if (isVerbose) {
                    System.out.printf(
                            "Episode %d: Total Reward=%.6f, Average Reward=%.6f, Total Steps=%d, Epsilon=%.6f, LearningRate=%.6f, Environment=%s, Q Values: %s, maxIndex = %.0f%n",
                            episode, cumulativeReward, cumulativeReward / dqnPath.size(), environment.getCurrentSteps(), agent.getEpsilon(), agent.getLearningRate(),
                            environment.getClass().getSimpleName(), qValues.toRowMatrix(), math.maxIndex(qValues).getY()
                    );
                }
            }
            dqnPath.add(environment.getAgentPosition());

            int pathLength = environment.getCurrentSteps();
            double meanReward = cumulativeReward / pathLength;

            if (isVerbose) {
                System.out.printf("Episode %d: Total Reward=%f, Average Reward=%f, Total Steps=%d, Epsilon=%f, LearningRate=%f, Environment=%s %n",
                        episode, cumulativeReward, meanReward, pathLength, agent.getEpsilon(), agent.getLearningRate(), environment.getClass().getSimpleName()
                );
            }
            if (dumpInfo) {
                agent.dumpDQNInfo();
            }

            // Progress Tracking -------------------------------------------------------------

            if (plot) {
                averageRewardPlotter.addPoint(new Vector2(episode, meanReward));
                if (tdErrorCounter != 0) averageLossPlotter.addPoint(new Vector2(episode, totalSquaredTDError / tdErrorCounter));
            }

            if (episode % savePeriod == 0) {
                agent.saveAgent("agent_" + episode + ".dat");
            }

            if (episode % visualiserUpdatePeriod == 0) {
                if (showPath && (visualiser == null || visualiser2 == null)) {
                    visualiser = new Environment_Visualiser(environment);
                    visualiser2 = new Environment_Visualiser(environment);
                }

                if (plot) {
                    averageRewardPlotter.plot();
                    averageLossPlotter.plot();
                }

                if (showPath) {
                    visualiser.reset(environment);
                    visualiser2.reset(environment);
                    visualiser.clearPaths();
                    visualiser2.clearPaths();
                    visualiser.addPath(Pathfinder.dijkstra(environment.getStartPosition(), environment.getGoalPosition(), environment), Color.ORANGE);
                    visualiser2.addPath(new ArrayList<>(dqnPath), Color.ORANGE);
                }
            }
        }
    }
}
