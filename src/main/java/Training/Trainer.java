package Training;

import Structures.DQNAgent;
import Structures.Matrix;
import Structures.Vector2D;
import Tools.Environment_Visualiser;
import Tools.GraphPlotter;
import Tools.Pathfinding.Pathfinder;
import Tools.math;
import com.sun.jdi.InvalidTypeException;

import javax.swing.*;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class Trainer {
    Set<Class<? extends GridEnvironment>> environmentClasses;

    public Trainer(Set<Class<? extends GridEnvironment>> environments) throws InvalidTypeException {
        this.environmentClasses = environments;
    }

    public void trainAgent(DQNAgent agent, int numEpisodes, int savePeriod, String... varargs) {
        List<String> args = Arrays.asList(varargs);
        boolean verbose = args.contains("verbose");

        boolean plot = args.contains("plot");
        GraphPlotter plotter = null;
        List<Vector2D> points = null;

        if (plot) {
            points = new ArrayList<>();

            plotter = new GraphPlotter("DQN training", GraphPlotter.Types.LINE, "Episode", "Cumulative Reward", points, varargs);
            plotter.setVisible(true);
            plotter.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        }

        boolean showPath = args.contains("show_path");
        Environment_Visualiser visualiser;  // for showing the path

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

        for (int episode = 1; episode <= numEpisodes; episode++) {
            GridEnvironment environment = environments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            System.out.println("Episode " + episode + ", environment: "+environment.getType());
            System.out.println("Start: " + environment.getStartPosition() + ", End: "+environment.getGoalPosition());

            Matrix state = environment.getState();
            boolean done;
            float totalReward = 0;
            ArrayList<Vector2D> dqnPath = new ArrayList<>();

            do {
                int action = agent.chooseAction(state);
                dqnPath.add(environment.getAgentPosition());
                Environment.MoveResult result = environment.step(action);
                agent.train(state, action, result.reward, result.state, result.done);
                state = result.state;
                done = result.done;
                totalReward += result.reward;
            } while (!done);

            if (plot) points.add(new Vector2D(episode, totalReward));

            if (episode % savePeriod == 0) {
                System.out.println("Episode " + episode + ": Total Reward = " + totalReward +", Total Steps = "+dqnPath.size());
                agent.saveAgent("agent_" + episode + ".dat");

                if (plot) {
                    plotter.addPoint(new Vector2D(episode, totalReward));
                }
                if (showPath) {
                    visualiser = new Environment_Visualiser(environment);
                    visualiser.addPath(Pathfinder.dijkstra(environment.getStartPosition(), environment.getGoalPosition(), environment), java.awt.Color.RED);
                }
            }
        }
    }
}
