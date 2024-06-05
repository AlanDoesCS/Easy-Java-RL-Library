package Tools;

import Structures.Matrix;
import Structures.Vector2D;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class PerlinNoise {
    Map<Integer, Float> gradients = new HashMap<>();
    // Map<int y, Map<int x, Vector2D gradient>> gradients2D
    Map<Integer, Map<Integer, Vector2D>> gradients2D = new HashMap<>();
    Random random = new Random();

    public static float lerp(float t, float a, float b) { // t is between 0 and 1
        return a + t * ( b - a );
    }

    static float ease(float x) { // easeInOut function between 0 and 1
        if (x<0) {
            return 0;
        } else if (x>1) {
            return 1;
        } else {
            return x < 0.5 ? 4 * x * x * x : (float) (1 - Math.pow(-2 * x + 2, 3) / 2);
        }
    } // from: https://easings.net/#easeInOutCubic

    static float ease(float x, float y) {
        return ease(x) * ease(y);
    }

    float rand(float lower, float upper) {
        return lower + random.nextFloat() * (upper - lower);
    }

    Vector2D randV(float lower, float upper) {
        return new Vector2D(rand(lower, upper), rand(lower, upper));

    }

    private float getGradient(int key) {
        if (!gradients.containsKey(key)) gradients.put(key, rand(-1, 1)); ;
        return gradients.get(key);
    }

    private Vector2D getGradient(int x, int y) {
        if (!gradients2D.containsKey(y)) gradients2D.put(y, new HashMap<>());
        Map<Integer, Vector2D> rowGradientMap = gradients2D.get(y);

        if (!rowGradientMap.containsKey(x)) {
            rowGradientMap.put(x, randV(-1, 1));
            gradients2D.put(y, rowGradientMap);
        }

        return rowGradientMap.get(x);
    }

    float[][] generateOctaves(int octaves, float persistence) {
        // Amplitude and frequency arrays
        float[] frequencies = new float[octaves];
        float[] amplitudes = new float[octaves];

        // Formula for frequency = 2^i, for amplitude = persistence^i, where i is the index
        for (int i=0; i<octaves; i++) {
            frequencies[i] = (float) Math.pow(2, i);
            amplitudes[i] = (float) Math.pow(persistence, i);
        }

        return new float[][]{frequencies, amplitudes};
    }

    public float noise(float x, int octaves, float persistence) {
        float noise = 0.0f;

        float[][] octaveArr = generateOctaves(octaves, persistence);
        float[] frequencies = octaveArr[0];
        float[] amplitudes = octaveArr[1];

        for (int i=0; i<octaves; i++) {
            float frequency = frequencies[i];
            float amplitude = amplitudes[i];
            float xPos = x * frequency;

            // Bounds
            int lower = (int) Math.floor(xPos); // round towards negative infinity
            int upper = lower+1;

            // Distances
            float d1 = xPos - lower;
            float d2 = xPos - upper;

            // Gradients
            float g1 = getGradient(lower);
            float g2 = getGradient(upper);

            // Noise vectors (dot product of distance and gradient vectors)
            float v1 = d1*g1;
            float v2 = d2*g2;

            noise += lerp(ease(d1), v1, v2)*amplitude;
        }
        return noise;
    }

    public float noise (float x, float y, int octaves, float persistence) {
        float noise = 0.0f;

        // amplitudes and frequencies
        float[][] octaveArr = generateOctaves(octaves, persistence);
        float[] frequencies = octaveArr[0];
        float[] amplitudes = octaveArr[1];



        for (int i=0; i<octaves; i++) {
            float frequency = frequencies[i];
            float amplitude = amplitudes[i];
            float xPos = x * frequency;
            float yPos = y * frequency;

            Vector2D position = new Vector2D(xPos, yPos);

            // Bounds
            int x0 = (int) Math.floor(xPos);
            int x1 = x0+1;
            int y0 = (int) Math.floor(xPos);
            int y1 = y0+1;

            // Distances
            Vector2D d0 = Vector2D.subtract(position, new Vector2D(x0, y0));
            Vector2D d1 = Vector2D.subtract(position, new Vector2D(x1, y0));
            Vector2D d2 = Vector2D.subtract(position, new Vector2D(x0, y1));
            Vector2D d3 = Vector2D.subtract(position, new Vector2D(x1, y1));

            // Gradients
            Vector2D g0 = getGradient(x0, y0);
            Vector2D g1 = getGradient(x1, y0);
            Vector2D g2 = getGradient(x0, y1);
            Vector2D g3 = getGradient(x1, y1);

            // Noise vectors (delta, scalar displacements)
            float v0 = Vector2D.dot(d0, g0);
            float v1 = Vector2D.dot(d1, g1);
            float v2 = Vector2D.dot(d2, g2);
            float v3 = Vector2D.dot(d3, g3);

            noise += (
                    ease(1-xPos, 1-yPos)*v0 +
                    ease(xPos, 1-yPos)*v1 +
                    ease(1-xPos, yPos)*v2 +
                    ease(xPos, yPos)*v3
                    );
        }
        return noise;
    }

    public Set<Integer> getKeySet() {
        return gradients.keySet();
    }

    public Matrix toMatrix(int xPixels, int yPixels, float step, int octaves, float persistence) {
        Matrix M = new Matrix(yPixels, xPixels);

        float yOffset = 0;
        for (int y=0; y<yPixels; y++) {
            float xOffset = 0;
            for (int x=0; x<xPixels; x++) {
                M.set(x, y, noise(xOffset, yOffset, octaves, persistence));
                xOffset += step;
            }
            yOffset += step;
        }
        return M;
    }
}
