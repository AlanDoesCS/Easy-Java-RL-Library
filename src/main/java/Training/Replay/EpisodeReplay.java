package Training.Replay;

import java.util.ArrayList;
import java.util.List;

public class EpisodeReplay {
    private List<Episode> buffer;
    private int capacity;

    public EpisodeReplay(int capacity) {
        this.capacity = capacity;
        this.buffer = new ArrayList<>(capacity);
    }

    public void add(Episode episode) {
        if (buffer.size() >= capacity) {
            buffer.removeFirst();
        }
        buffer.add(episode);
    }

    public Episode sample() {
        return buffer.get((int) (Math.random() * buffer.size()-1));
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

    public static class Episode {
        public List<ExperienceReplay.Experience> experiences;
        public double totalReward;

        public Episode() {
            experiences = new ArrayList<>();
            totalReward = 0;
        }

        public void addExperience(ExperienceReplay.Experience exp) {
            experiences.add(exp);
            totalReward += exp.reward;
        }
    }
}
