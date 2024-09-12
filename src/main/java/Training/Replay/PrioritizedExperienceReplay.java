package Training.Replay;

import java.util.*;

public class PrioritizedExperienceReplay {
    private SumTree tree;
    private int capacity;
    private double epsilon = 0.01f;
    private double alpha = 0.6f;
    private double beta = 0.4f;
    private double betaIncrement = 0.001f;
    private double maxPriority = 1.0f;

    // Tree structure for priority sampling
    private static class SumTree {
        private double[] tree;
        private ExperienceReplay.Experience[] data;
        private int capacity;
        private int count;
        private int dataPointer;

        public SumTree(int capacity) {
            this.capacity = capacity;
            this.tree = new double[2 * capacity - 1];
            this.data = new ExperienceReplay.Experience[capacity];
            this.count = 0;
            this.dataPointer = 0;
        }

        public synchronized void add(double priority, ExperienceReplay.Experience experience) {
            int treeIndex = this.dataPointer + this.capacity - 1;
            if (this.data[this.dataPointer] != null) {
                this.update(treeIndex, 0);
            }
            this.data[this.dataPointer] = experience;
            experience.index = treeIndex;
            this.update(treeIndex, priority);

            this.dataPointer = (this.dataPointer + 1) % this.capacity;
            if (this.count < this.capacity) this.count++;
        }

        public synchronized void update(int treeIndex, double priority) {
            double change = priority - this.tree[treeIndex];
            this.tree[treeIndex] = priority;
            while (treeIndex != 0) {
                treeIndex = (treeIndex - 1) / 2;
                this.tree[treeIndex] += change;
            }
        }

        public synchronized Sample get(double s) {
            if (this.count == 0) {
                throw new IllegalStateException("Attempting to get from empty SumTree");
            }

            int parentIndex = 0;
            while (true) {
                int leftChildIndex = 2 * parentIndex + 1;
                int rightChildIndex = leftChildIndex + 1;

                if (leftChildIndex >= this.tree.length) {
                    break;
                }

                if (s <= this.tree[leftChildIndex]) {
                    parentIndex = leftChildIndex;
                } else {
                    s -= this.tree[leftChildIndex];
                    parentIndex = rightChildIndex;
                }
            }

            int dataIndex = parentIndex - this.capacity + 1;

            if (dataIndex < 0 || dataIndex >= this.capacity) {
                throw new IllegalStateException("Invalid data index in SumTree.get: " + dataIndex);
            }

            return new Sample(parentIndex, this.tree[parentIndex], this.data[dataIndex]);
        }

        public synchronized double total() {
            return this.tree[0];
        }
    }

    public boolean hasEnoughSamples(int batchSize) {
        return this.tree.count >= batchSize;
    }

    private static class Sample {
        int treeIndex;
        double priority;
        ExperienceReplay.Experience experience;

        public Sample(int treeIndex, double priority, ExperienceReplay.Experience experience) {
            this.treeIndex = treeIndex;
            this.priority = priority;
            this.experience = experience;
        }
    }

    public PrioritizedExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.tree = new SumTree(capacity);
    }

    public synchronized void setCapacity(int newCapacity) {
        if (newCapacity < this.capacity) {
            throw new IllegalArgumentException("New capacity must be greater than or equal to current capacity.");
        }
        SumTree newTree = new SumTree(newCapacity);
        for (int i = 0; i < this.tree.count; i++) {
            ExperienceReplay.Experience experience = this.tree.data[i];
            double priority = this.tree.tree[i + this.tree.capacity - 1];
            newTree.add(priority, experience);
        }
        this.capacity = newCapacity;
        this.tree = newTree;
    }

    public synchronized void add(ExperienceReplay.Experience experience) {
        double priority = Math.max(this.epsilon, this.maxPriority);
        this.tree.add(priority, experience);
    }

    public synchronized List<ExperienceReplay.Experience> sample(int batchSize) {
        if (!hasEnoughSamples(batchSize)) {
            throw new IllegalStateException("Not enough samples in buffer. Current size: " + this.tree.count + ", Required: " + batchSize);
        }

        List<ExperienceReplay.Experience> batch = new ArrayList<>();
        double segment = this.tree.total() / batchSize;

        this.beta = Math.min(1.0f, this.beta + this.betaIncrement);

        for (int i = 0; i < batchSize; i++) {
            double a = segment * i;
            double b = segment * (i + 1);
            double s = Math.random() * (b - a) + a;
            Sample sample = this.tree.get(s);
            batch.add(sample.experience);
        }

        return batch;
    }

    public synchronized void updatePriorities(List<Integer> treeIndices, List<Double> tdErrors) {
        for (int i = 0; i < treeIndices.size(); i++) {
            double priority = Math.pow(Math.abs(tdErrors.get(i)) + this.epsilon, this.alpha);
            this.tree.update(treeIndices.get(i), priority);
            this.maxPriority = Math.max(this.maxPriority, priority);
        }
    }

    public int getCapacity() {
        return capacity;
    }

    public synchronized int size() {
        return this.tree.count;
    }
}
