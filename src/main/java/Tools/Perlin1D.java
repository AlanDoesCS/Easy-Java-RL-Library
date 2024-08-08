package Tools;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Perlin1D extends PerlinNoise {
    Map<Integer, Float> gradients = new HashMap<>();
    Random random = new Random();

    public Perlin1D(int octaves, float persistence) {
        super(octaves, persistence);
    }

    private float getGradient(int key) {
        if (!gradients.containsKey(key)) gradients.put(key, rand(-1, 1)); ;
        return gradients.get(key);
    }

    public float noise(float x) {
        float noise = 0.0f;

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

            noise += lerp(fade(d1), v1, v2)*amplitude;
        }
        return noise;
    }

    public float noise(float x, float target_min, float target_max) {
        return math.scale(noise(x), -1, 1, target_min, target_max);
    }
}
