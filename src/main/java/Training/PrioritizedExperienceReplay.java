package Training;

public class PrioritizedExperienceReplay {
    private SumTree tree;
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
            this.data[this.dataPointer] = experience;
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
            return new Sample(parentIndex, this.tree[parentIndex], this.data[dataIndex]);
        }

        public float total() {
            return this.tree[0];
        }
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
}
