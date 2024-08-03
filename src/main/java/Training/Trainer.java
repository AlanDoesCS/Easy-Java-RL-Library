package Training;

import Structures.DQNAgent;
import Structures.Matrix;
import Tools.Environment_Visualiser;
import Tools.math;
import com.sun.jdi.InvalidTypeException;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class Trainer {
    Set<Class<? extends Environment>> environmentClasses;

    public Trainer(Set<Class<? extends Environment>> environments) throws InvalidTypeException {
        this.environmentClasses = environments;
    }

    public void trainAgent(DQNAgent agent, int numEpisodes, int savePeriod) {
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
            boolean done = false;
            float totalReward = 0;

            while (!done) {
                int action = agent.chooseAction(state);
                Environment.MoveResult result = environment.step(action);
                agent.train(state, action, result.reward, result.state, result.done);
                state = result.state;
                done = result.done;
                totalReward += result.reward;
            }

            if (episode % savePeriod == 0) {
                System.out.println("Episode " + episode + ": Total Reward = " + totalReward);
                agent.saveAgent("agent_" + episode + ".dat");
            }
        }
    }
}
