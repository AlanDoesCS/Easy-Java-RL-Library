package Tools;

import java.util.Random;

public class math {
    public static float Q_rsqrt(float x, int precision) { // from Quake III
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        for (int iter = 0; iter < precision; ++iter) { // precision of 3 seems sufficient
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
}
