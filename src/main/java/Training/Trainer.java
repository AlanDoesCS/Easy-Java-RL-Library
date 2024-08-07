package Training;

import Structures.DQNAgent;
import Structures.Matrix;
import Structures.Vector2D;
import Tools.GraphPlotter;
import Tools.math;
import com.sun.jdi.InvalidTypeException;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

public class Trainer {
    Set<Class<? extends Environment>> environmentClasses;

    public Trainer(Set<Class<? extends Environment>> environments) throws InvalidTypeException {
        this.environmentClasses = environments;
    }

    public void trainAgent(DQNAgent agent, int numEpisodes, int savePeriod, String... varargs) {
        List<String> args = Arrays.asList(varargs);
        boolean plot = args.contains("plot");

        GraphPlotter plotter;
        List<Vector2D> points = null;

        if (plot) {
            points = new ArrayList<>();
        }

        List<Environment> environments = new ArrayList<>();

        try {
            for (Class<? extends Environment> envClass : environmentClasses) {
                environments.add(Environment.of(envClass));
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        for (int episode = 1; episode <= numEpisodes; episode++) {
            Environment environment = environments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            System.out.println("Episode " + episode + ", environment: "+environment.getType());

            Matrix state = environment.getState();
            boolean done;
            float totalReward = 0;

            do {
                int action = agent.chooseAction(state);
                Environment.MoveResult result = environment.step(action);
                agent.train(state, action, result.reward, result.state, result.done);
                state = result.state;
                done = result.done;
                totalReward += result.reward;
            } while (!done);

            if (plot) points.add(new Vector2D(episode, totalReward));

            if (episode % savePeriod == 0) {
                System.out.println("Episode " + episode + ": Total Reward = " + totalReward);
                agent.saveAgent("agent_" + episode + ".dat");

                if (plot) {
                    plotter = new GraphPlotter("DQN training, ep: "+episode, GraphPlotter.Types.LINE, "Episode", "Cumulative Reward", points, varargs);
                    plotter.setVisible(true);
                    plotter.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
                }
            }
        }
    }
}
