package Tools;

import java.util.Random;

public abstract class PerlinNoise {
    final int octaves;
    final float persistence;
    final float[] frequencies, amplitudes;
    Random random = new Random();

    public PerlinNoise(int octaves, float persistence) {
        this.octaves = octaves;
        this.persistence = persistence;
        float[][] octavesArray = generateOctaves(octaves, persistence);
        frequencies = octavesArray[0];
        amplitudes = octavesArray[1];
    }

    public static float lerp(float t, float a, float b) { // t is between 0 and 1
        return a + t * ( b - a );
    }

    static float fade(float t) { // fade function (ψ)
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    float rand(float lower, float upper) {
        return lower + random.nextFloat() * (upper - lower);
    }

    static float[][] generateOctaves(int octaves, float persistence) {
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
}
