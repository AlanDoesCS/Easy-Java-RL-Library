package Tools;

import Structures.MatrixDouble;
import Structures.Vector2;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Perlin2D extends PerlinNoise {
    Map<Integer, Map<Integer, Vector2>> gradients2D = new HashMap<>();
    Random random = new Random();
    Vector2 randomOffset = new Vector2(random.nextFloat(), random.nextFloat());

    public Perlin2D(int octaves, float persistence) {
        super(octaves, persistence);
    }

    static float fade(float x, float y) { // Ψ
        return fade(x) * fade(y);
    }

    private Vector2 getGradient(int x, int y) {
        if (!gradients2D.containsKey(y)) gradients2D.put(y, new HashMap<>());
        Map<Integer, Vector2> rowGradientMap = gradients2D.get(y);

        if (!rowGradientMap.containsKey(x)) {
            rowGradientMap.put(x, Vector2.randomUnitVect(random));
            gradients2D.put(y, rowGradientMap);
        }

        return rowGradientMap.get(x);
    }

    public float noise(float x, float y) {
        float noise = 0.0f;

        for (int i=0; i<octaves; i++) {
            float frequency = frequencies[i];
            float amplitude = amplitudes[i];
            float xPos = (float) ((x+randomOffset.getX()) * frequency);
            float yPos = (float) ((y+randomOffset.getY()) * frequency);

            Vector2 position = new Vector2(xPos, yPos);

            // Bounds
            int x0 = (int) Math.floor(xPos);
            int x1 = x0+1;
            int y0 = (int) Math.floor(yPos);
            int y1 = y0+1;

            // Distances
            Vector2 v0 = Vector2.subtract(position, new Vector2(x0, y0));
            Vector2 v1 = Vector2.subtract(position, new Vector2(x1, y0));
            Vector2 v2 = Vector2.subtract(position, new Vector2(x0, y1));
            Vector2 v3 = Vector2.subtract(position, new Vector2(x1, y1));

            // Vertical displacement (delta)
            float d0 = (float) Vector2.dot(v0, getGradient(x0, y0));
            float d1 = (float) Vector2.dot(v1, getGradient(x1, y0));
            float d2 = (float) Vector2.dot(v2, getGradient(x0, y1));
            float d3 = (float) Vector2.dot(v3, getGradient(x1, y1));

            float xf = xPos -x0;
            float yf = yPos -y0;
            noise += (
                    fade(1 - xf, 1 - yf) * d0 +
                            fade(xf, 1 - yf) * d1 +
                            fade(1 - xf, yf) * d2 +
                            fade(xf, yf) * d3
            ) * amplitude;
        }
        return math.clamp(noise, -1f, 1f);
        //return Math.round(noise*5)/5f; // For "stepped" results
    }

    public float noise(float x, float y, float range_min, float range_max) {
        return math.scale(noise(x, y), -1, 1, range_min, range_max);
    }

    public MatrixDouble toMatrix(int xPixels, int yPixels, float step) {
        MatrixDouble M = new MatrixDouble(yPixels, xPixels);

        float yOffset = 0;
        for (int y=0; y<yPixels; y++) {
            float xOffset = 0;
            for (int x=0; x<xPixels; x++) {
                M.set(x, y, noise(xOffset, yOffset));
                xOffset += step;
            }
            yOffset += step;
        }
        return M;
    }
}
