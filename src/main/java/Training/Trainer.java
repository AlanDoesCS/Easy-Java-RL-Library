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
        GraphPlotter plotter = null;

        if (plot) {
            plotter = new GraphPlotter("DQN training", GraphPlotter.Types.LINE, "Episode", "Cumulative Reward", varargs);
            plotter.setVisible(true);
            plotter.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        }

        boolean showPath = args.contains("show_path");
        Environment_Visualiser visualiser;  // for showing the path - Dijkstra's path
        Environment_Visualiser visualiser2; // for showing the path - DQN's path

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

        EpisodeReplay replay = new EpisodeReplay(1000);
        int batchSize = 64;
        int updatePeriod = 4;
        int totalSteps;

        for (int episode = 1; episode <= numEpisodes; episode++) {
            EpisodeReplay.Episode currentEpisode = new EpisodeReplay.Episode();

            GridEnvironment environment = environments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            Tensor state = environment.getState();
            boolean done = false;
            float totalReward = 0;
            ArrayList<Vector2D> dqnPath = new ArrayList<>();

            totalSteps = 0;
            while (!done) {
                int action = agent.chooseAction(state);
                dqnPath.add(environment.getAgentPosition());
                Environment.MoveResult result = environment.step(action);

                totalSteps++;

                // Add experience to current episode
                currentEpisode.addExperience(new ExperienceReplay.Experience(state, action, result.reward, result.state, result.done));

                state = result.state;
                done = result.done;
                totalReward += result.reward;

                if (verbose) System.out.printf("Current: %s, Goal:%s, Step reward:%f, Total reward:%f%n", environment.getAgentPosition(), environment.getGoalPosition(), result.reward, totalReward);
            }

            replay.add(currentEpisode);
            System.out.println("\nEpisode " + episode + ": Total Reward = " + totalReward +", Total Steps = "+dqnPath.size()+", Environment = "+environment.getClass().getSimpleName()+"\n");

            if (replay.size() > 0) {
                EpisodeReplay.Episode trainingEpisode = replay.sample();
                for (ExperienceReplay.Experience exp : trainingEpisode.experiences) {
                    agent.train(exp.state, exp.action, exp.reward, exp.nextState, exp.done);
                }
            }

            // Progress Tracking -------------------------------------------------------------

            if (plot) plotter.addPoint(new Vector2D(episode, totalReward));

            if (episode % savePeriod == 0) {
                System.out.println("Episode " + episode + ": Total Reward = " + totalReward +", Total Steps = "+dqnPath.size());
                agent.saveAgent("agent_" + episode + ".dat");
            }

            if (episode % visualiserUpdatePeriod == 0) {
                if (plot) plotter.plot();

                if (showPath) {
                    visualiser = new Environment_Visualiser(environment);
                    visualiser.addPath(Pathfinder.dijkstra(environment.getStartPosition(), environment.getGoalPosition(), environment), java.awt.Color.RED);
                    visualiser2 = new Environment_Visualiser(environment);
                    visualiser2.addPath(dqnPath, Color.GREEN);
                }
            }
        }
    }
}
