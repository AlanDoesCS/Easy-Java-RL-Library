package Structures;

/**
 * CircularBuffer is a data structure that maintains a fixed-size buffer
 * and supports operations to add elements, calculate statistics, and
 * retrieve buffer properties.
 */
public class CircularBuffer {
    private final double[] buffer;
    private int currentSize;
    private int head;
    private int tail;
    private double sum;
    private double sumOfSquares;
    private double min;
    private double max;

    /**
     * Constructs a CircularBuffer with the specified capacity.
     *
     * @param capacity the maximum number of elements the buffer can hold
     */
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

    /**
     * Adds a value to the buffer. If the buffer is full, the oldest value is removed.
     *
     * @param value the value to add to the buffer
     */
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

    /**
     * Returns the mean of the values in the buffer.
     *
     * @return the mean of the buffer values
     * @throws IllegalStateException if the buffer is empty
     */
    public double getMean() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return sum / currentSize;
    }

    /**
     * Returns the minimum value in the buffer.
     *
     * @return the minimum value in the buffer
     * @throws IllegalStateException if the buffer is empty
     */
    public double getMin() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return min;
    }

    /**
     * Returns the maximum value in the buffer.
     *
     * @return the maximum value in the buffer
     * @throws IllegalStateException if the buffer is empty
     */
    public double getMax() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return max;
    }

    /**
     * Returns the standard deviation of the values in the buffer.
     *
     * @return the standard deviation of the buffer values
     * @throws IllegalStateException if the buffer is empty
     */
    public double getStandardDeviation() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        double mean = getMean();
        return Math.sqrt((sumOfSquares / currentSize) - (mean * mean));
    }

    /**
     * Returns the variance of the values in the buffer.
     *
     * @return the variance of the buffer values
     * @throws IllegalStateException if the buffer is empty
     */
    public double getVariance() {
        if (currentSize == 0) {
            throw new IllegalStateException("Buffer is empty");
        }
        return (sumOfSquares / currentSize) - Math.pow(getMean(), 2);
    }

    /**
     * Returns the current number of elements in the buffer.
     *
     * @return the current size of the buffer
     */
    public int getSize() {
        return currentSize;
    }

    /**
     * Returns the maximum capacity of the buffer.
     *
     * @return the capacity of the buffer
     */
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
