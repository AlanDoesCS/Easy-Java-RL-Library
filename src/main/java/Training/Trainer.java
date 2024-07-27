package Training;

import java.util.List;

public class Trainer {
    List<Class<? extends Environment>> environments;

    public Trainer(List<Class<? extends Environment>> environments) {
        this.environments = environments;
    }
}
