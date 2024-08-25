package Structures;

public class CircularBuffer {
    private final double[] buffer;
    private int currentSize;
    private int head;
    private int tail;
    private double sum;

    public CircularBuffer(int capacity) {
        buffer = new double[capacity];
        currentSize = 0;
        head = 0;
        tail = 0;
        sum = 0;
    }

    public void add(double value) {
        if (currentSize == buffer.length) {
            sum -= buffer[tail];
            tail = (tail + 1) % buffer.length;
        } else {
            currentSize++;
        }

        buffer[head] = value;
        sum += value;
        head = (head + 1) % buffer.length;
    }

    public double getAverage() {
        if (currentSize == 0) {
            return 0;
        }
        return sum / currentSize;
    }

    public int getSize() {
        return currentSize;
    }

    public int getCapacity() {
        return buffer.length;
    }
}
