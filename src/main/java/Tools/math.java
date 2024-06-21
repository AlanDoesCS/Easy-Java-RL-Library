package Tools;

import Tools.Pathfinding.GraphNode;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;

public class math {
    public static float Q_rsqrt(float x, int precision) { // from Quake III
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        for (int iter = 0; iter < precision; ++iter) { // precision of 3 seems sufficient in most cases
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
    public static float percentAccuracy(float predictedWeight, float actualWeight) {
        return (1 - Math.abs(actualWeight-predictedWeight)/actualWeight) * 100;
    }

    /*
    MSE is calculated using the weights of each visited node, sorted in ascending order
     */
    public static float MSE(ArrayList<GraphNode> predictedPath, ArrayList<GraphNode> actualPath) {
        return -1;
    }

    public static float clamp(float noise, float min, float max) {
        if (noise < min) return min;
        return Math.min(noise, max);
    }

}
