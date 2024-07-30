package Training;

import Structures.DQNAgent;
import Tools.math;

import java.util.ArrayList;
import java.util.List;

public class Trainer {
    List<Class<? extends Environment>> environmentClasses;

    public Trainer(List<Class<? extends Environment>> environments) {
        this.environmentClasses = environments;
    }

    public void trainAgent(DQNAgent agent, int numEpisodes, int savePeriod, int environmentChangePeriod) {
        List<Environment> trainingEnvironments = new ArrayList<>();

        try {
            for (Class<? extends Environment> envClass : environmentClasses) {
                trainingEnvironments.add(envClass.getDeclaredConstructor().newInstance());
            }
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        for (int episode = 0; episode < numEpisodes; episode++) {
            Environment environment = trainingEnvironments.get(math.randomInt(0, environmentClasses.size()-1));
            environment.randomize();

            // TODO: proper training
        }
    }
}
