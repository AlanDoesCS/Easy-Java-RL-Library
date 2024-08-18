package Training;

import Tools.math;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

public class PrioritizedExperienceReplay {
    private SumTree tree;
    private int capacity;
    private float epsilon = 0.01f;
    private float alpha = 0.6f;
    private float beta = 0.4f;
    private float betaIncrement = 0.001f;
    private float maxPriority = 1.0f;

    private static class SumTree {
        private float[] tree;
        private ExperienceReplay.Experience[] data;
        private int capacity;
        private int count;
        private int dataPointer;

        public SumTree(int capacity) {
            this.capacity = capacity;
            this.tree = new float[2 * capacity - 1];
            this.data = new ExperienceReplay.Experience[capacity];
            this.count = 0;
            this.dataPointer = 0;
        }

        public void add(float priority, ExperienceReplay.Experience experience) {
            int treeIndex = this.dataPointer + this.capacity - 1;
            if (Objects.isNull(experience)) {
                throw new IllegalStateException("Null experience passed into SumTree.add");
            }
            if (experience.state == null) {
                throw new IllegalStateException("Experience with null state passed into SumTree.add");
            }
            if (experience.nextState == null) {
                throw new IllegalStateException("Experience with null nextState passed into SumTree.add");
            }
            if (this.data[this.dataPointer] != null) {
                // Remove the old experience from the tree
                this.update(treeIndex, 0);
            }
            this.data[this.dataPointer] = experience;
            experience.index = treeIndex;
            this.update(treeIndex, priority);

            this.dataPointer = (this.dataPointer + 1) % this.capacity;
            if (this.count < this.capacity) this.count++;
        }

        public void update(int treeIndex, float priority) {
            float change = priority - this.tree[treeIndex];
            this.tree[treeIndex] = priority;
            while (treeIndex != 0) {
                treeIndex = (treeIndex - 1) / 2;
                this.tree[treeIndex] += change;
            }
        }

        public Sample get(float s) {
            if (this.count == 0) {
                System.err.println("Attempting to sample from an empty SumTree");
                return null;
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

            if (this.data[dataIndex] == null) {
                throw new IllegalStateException("Null experience at valid index in SumTree.get: " + dataIndex);
            }

            return new Sample(parentIndex, this.tree[parentIndex], this.data[dataIndex]);
        }

        public float total() {
            return this.tree[0];
        }
    }

    public boolean hasEnoughSamples(int batchSize) {
        return this.tree.count >= batchSize;
    }

    private static class Sample {
        int treeIndex;
        float priority;
        ExperienceReplay.Experience experience;

        public Sample(int treeIndex, float priority, ExperienceReplay.Experience experience) {
            this.treeIndex = treeIndex;
            this.priority = priority;
            this.experience = experience;
        }
    }

    public PrioritizedExperienceReplay(int capacity) {
        this.capacity = capacity;
        this.tree = new SumTree(capacity);
    }

    public void add(ExperienceReplay.Experience experience) {
        if (experience == null) {
            throw new IllegalStateException("Attempting to add null experience to PrioritizedExperienceReplay");
        }
        float priority = Math.max(this.epsilon, this.maxPriority);
        if (Float.isNaN(priority)) {
            throw new IllegalStateException("NaN priority in PrioritizedExperienceReplay.add");
        }
        this.tree.add(priority, experience);
    }

    public List<ExperienceReplay.Experience> sample(int batchSize) {
        if (!hasEnoughSamples(batchSize)) {
            throw new IllegalStateException("Not enough samples in buffer. Current size: " + this.tree.count + ", Required: " + batchSize);
        }

        List<ExperienceReplay.Experience> batch = new ArrayList<>();
        List<Integer> treeIndices = new ArrayList<>();
        List<Float> priorities = new ArrayList<>();

        float segment = this.tree.total() / batchSize;
        this.beta = Math.min(1.0f, this.beta + this.betaIncrement);

        for (int i = 0; i < batchSize; i++) {
            float a = segment * i;
            float b = segment * (i + 1);
            float s = (float) Math.random() * (b - a) + a;
            Sample sample = this.tree.get(s);

            if (sample.experience == null) {
                System.err.println("Sampled null experience from PrioritizedExperienceReplay");
                continue;
            }

            batch.add(sample.experience);
            treeIndices.add(sample.treeIndex);
            priorities.add(sample.priority);
        }

        float maxWeight = (float) Math.pow(this.tree.total() / this.maxPriority, -this.beta);
        for (int i = 0; i < batchSize; i++) {
            float weight = (float) Math.pow(priorities.get(i) / this.tree.total(), -this.beta);
            weight = weight / maxWeight;
            priorities.set(i, weight);
        }

        return batch;
    }

    public void updatePriorities(List<Integer> treeIndices, List<Float> tdErrors) {
        for (int i = 0; i < treeIndices.size(); i++) {
            float priority = (float) Math.pow(Math.abs(tdErrors.get(i)) + this.epsilon, this.alpha);
            if (Float.isNaN(priority)) {
                throw new IllegalStateException("NaN priority in PrioritizedExperienceReplay.updatePriorities");
            }
            this.tree.update(treeIndices.get(i), priority);
            this.maxPriority = Math.max(this.maxPriority, priority);
        }
    }

    public int getCapacity() {
        return capacity;
    }

    public void setCapacity(int capacity) {
        this.capacity = capacity;
        // TODO: Implement rebuilding tree
    }

    public int size() {
        return this.tree.count;
    }
}
