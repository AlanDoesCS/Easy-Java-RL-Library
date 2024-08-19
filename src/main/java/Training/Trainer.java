package Training;

import Structures.DQNAgent;
import Structures.Tensor;
import Structures.Vector2D;
import Tools.Environment_Visualiser;
import Tools.GraphPlotter;
import Tools.Pathfinding.Pathfinder;
import Tools.math;
import com.sun.jdi.InvalidTypeException;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class Trainer {
    Set<Class<? extends GridEnvironment>> environmentClasses;

    public Trainer(Set<Class<? extends GridEnvironment>> environments) throws InvalidTypeException {
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
    public void trainAgent(DQNAgent agent, int numEpisodes, int savePeriod, int visualiserUpdatePeriod, String... varargs) {
        List<String> args = Arrays.asList(varargs);
        boolean verbose = args.contains("verbose");

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

        if (verbose) {
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

        PrioritizedExperienceReplay replay = new PrioritizedExperienceReplay(10000);
        int batchSize = 32;

        for (int episode = 1; episode <= numEpisodes; episode++) {
            float totalSquaredTDError = 0; int tdErrorCounter = 0;

            GridEnvironment environment = environments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            Tensor state = environment.getState();
            boolean done = false;
            float cumulativeReward = 0;
            ArrayList<Vector2D> dqnPath = new ArrayList<>();

            while (!done) {
                int action = agent.chooseAction(state);
                if (!dqnPath.isEmpty()) {
                    if (!dqnPath.getLast().equals(environment.getAgentPosition())) {
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
                    List<Float> tdErrors = new ArrayList<>();

                    for (ExperienceReplay.Experience exp : batch) {
                        float tdError = agent.train(exp.state, exp.action, exp.reward, exp.nextState, exp.done);
                        treeIndices.add(exp.index);
                        tdErrors.add(tdError);

                        totalSquaredTDError += tdError*tdError;
                        tdErrorCounter++;
                    }

                    replay.updatePriorities(treeIndices, tdErrors);
                }

                state = result.state;
                done = result.done;
                cumulativeReward += result.reward;

                if (verbose) System.out.printf("Current: %s, Goal:%s, Step reward:%f, Total reward:%f, Steps: (%d)%n", environment.getAgentPosition(), environment.getGoalPosition(), result.reward, cumulativeReward, environment.getCurrentSteps());
            }
            int pathLength = environment.getCurrentSteps();
            float meanReward = cumulativeReward / pathLength;

            System.out.printf("Episode %d: Total Reward=%f, Average Reward=%f, Total Steps=%d, Epsilon=%f, LearningRate=%f, Environment=%s %n",
                    episode, cumulativeReward, cumulativeReward / dqnPath.size(), dqnPath.size(), agent.getEpsilon(), agent.getLearningRate(), environment.getClass().getSimpleName()
            );
            agent.dumpDQNInfo();


            // Progress Tracking -------------------------------------------------------------

            if (plot) {
                averageRewardPlotter.addPoint(new Vector2D(episode, meanReward));
                averageLossPlotter.addPoint(new Vector2D(episode, totalSquaredTDError / tdErrorCounter));
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
