package Tools;

import Tools.Pathfinding.GraphNode;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;

public class math {
    static Random random = new Random();

    /**
     * Computes the fast inverse square root of a number using the Quake III algorithm.
     *
     * @param x the number to compute the inverse square root of
     * @param precision the number of newton iterations to done on the result; a precision of 3 is usually sufficient
     * @return the inverse square root of x
     */
    public static float Q_rsqrt(float x, int precision) {
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        for (int iter = 0; iter < precision; ++iter) {
            x *= (1.5f - xhalf * x * x);
        }
        return x;
    }
    public static float fastSqrt(float x, int precision) {
        return x * Q_rsqrt(x, precision);
    }
    public static float fastSqrt(float x) {
        return x * Q_rsqrt(x, 3);
    }
    public static float randomFloat(float min, float max, Random random) {
        return random.nextFloat() * (max - min) + min;
    }
    /**
     * Generates a random float within the specified range [min, max].
     *
     * @param min the minimum value of the range (inclusive)
     * @param max the maximum value of the range (inclusive)
     * @return a random float between min (inclusive) and max (inclusive)
     */
    public static float randomFloat(float min, float max) {
        return random.nextFloat() * (max - min) + min;
    }

    /**
     * Generates a random integer within the specified range [min, max].
     *
     * @param min the minimum value of the range (inclusive)
     * @param max the maximum value of the range (inclusive)
     * @return a random integer between min (inclusive) and max (inclusive)
     */
    public static int randomInt(int min, int max) {
        return random.nextInt((max - min) + 1) + min;
    }
    public static float random() {
        return random.nextFloat();
    }
    public static float percentAccuracy(float predictedWeight, float actualWeight) {
        return (1 - Math.abs(actualWeight-predictedWeight)/actualWeight) * 100;
    }

    public static float clamp(float value, float min, float max) {
        if (value < min) return min;
        return Math.min(value, max);
    }
    public static float lerp(float t, float a, float b) { // t is between 0 and 1
        return a + t * ( b - a );
    }

    /**
     * Normalises a value in range [from_min, from_max] to [0, 1]
     *
     * @param value the value to normalise
     * @param from_min the minimum value of the original range
     * @param from_max the maximum value of the original range
     * @return the normalised value in the range [0, 1]
     */
    public static float normalise(float value, float from_min, float from_max) {
        return (value - from_min)/(from_max - from_min);
    }

    /**
     * Scales a value from one range to another.
     *
     * @param value the value to scale
     * @param min_start the minimum value of the original range
     * @param max_start the maximum value of the original range
     * @param min_end the minimum value of the target range
     * @param max_end the maximum value of the target range
     * @return the scaled value in the target range
     */
    public static float scale(float value, float min_start, float max_start, float min_end, float max_end) {
        value = normalise(value, min_start, max_start);
        return value * (max_end - min_end) + min_end;
    }
}
