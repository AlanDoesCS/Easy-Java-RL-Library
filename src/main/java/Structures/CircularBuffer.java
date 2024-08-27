package Structures;

public class CircularBuffer {
    private final double[] buffer;
    private int currentSize;
    private int head;
    private int tail;
    private double sum;
    private double sumOfSquares;
    private double min;
    private double max;

    public CircularBuffer(int capacity) {
        buffer = new double[capacity];
        currentSize = 0;
        head = 0;
        tail = 0;
        sum = 0;
        sumOfSquares = 0;
        min = Double.MAX_VALUE;
        max = Double.MIN_VALUE;
    }

    public void add(double value) {
        if (currentSize == buffer.length) {
            double poppedVal = buffer[tail];
            sum -= poppedVal;
            sumOfSquares -= poppedVal * poppedVal;
            tail = (tail + 1) % buffer.length;

            if (poppedVal == min || poppedVal == max) {
                recalculateMinMax();
            }
        } else {
            currentSize++;
        }

        buffer[head] = value;
        sum += value;
        sumOfSquares += value * value;

        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }

        head = (head + 1) % buffer.length;
    }

    public double getMean() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return sum / currentSize;
    }

    public double getMin() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return min;
    }

    public double getMax() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return max;
    }

    public double getStandardDeviation() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        double mean = getMean();
        return Math.sqrt((sumOfSquares / currentSize) - (mean * mean));
    }

    public double getVariance() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return (sumOfSquares / currentSize) - Math.pow(getMean(), 2);
    }

    public int getSize() {
        return currentSize;
    }

    public int getCapacity() {
        return buffer.length;
    }

    private void recalculateMinMax() {
        min = Double.MAX_VALUE;
        max = Double.MIN_VALUE;

        for (int i = 0; i < currentSize; i++) {
            double value = buffer[(tail + i) % buffer.length];
            if (value < min) {
                min = value;
            }
            if (value > max) {
                max = value;
            }
        }
    }
}
