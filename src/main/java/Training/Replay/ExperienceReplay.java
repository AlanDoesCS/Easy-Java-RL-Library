package Training.Replay;

import Tools.math;
import java.util.ArrayList;
import java.util.List;

public class ExperienceReplay {
    private List<Experience> buffer;
    private int capacity;

    public ExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
    }

    // Add experiences to the buffer, removing the oldest if capacity is reached
    public void add(Experience experience) {
        if (buffer.size() >= capacity) {
            buffer.removeFirst();  // Removes the first experience (FIFO)
        }
        buffer.add(experience);
    }

    // Sample a batch of experiences randomly
    public List<Experience> sample(int batchSize) {
        List<Experience> batch = new ArrayList<>(batchSize);
        for (int i = 0; i < batchSize; i++) {
            batch.add(buffer.get(math.randomInt(0, buffer.size() - 1)));
        }
        return batch;
    }

    public int getCapacity() {
        return capacity;
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
    }

    public int size() {
        return buffer.size();
    }

    public static class Experience {
        public Object state;
        public int action;
        public double reward;
        public Object nextState;
        public boolean done;
        public int index;

        public Experience(Object state, int action, double reward, Object nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }

        @Override
        public String toString() {
            return "Exp(" + state + ", " + action + ", " + reward + ", " + nextState + ", " + done + ", " + index + ")";
        }
    }
}
