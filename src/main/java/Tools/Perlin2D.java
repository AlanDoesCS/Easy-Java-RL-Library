package Tools;

import Structures.MatrixDouble;
import Structures.Vector2D;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class Perlin2D extends PerlinNoise {
    Map<Integer, Map<Integer, Vector2D>> gradients2D = new HashMap<>();
    Random random = new Random();
    Vector2D randomOffset = new Vector2D(random.nextFloat(), random.nextFloat());

    public Perlin2D(int octaves, float persistence) {
        super(octaves, persistence);
    }

    static float fade(float x, float y) { // Ψ
        return fade(x) * fade(y);
    }

    private Vector2D getGradient(int x, int y) {
        if (!gradients2D.containsKey(y)) gradients2D.put(y, new HashMap<>());
        Map<Integer, Vector2D> rowGradientMap = gradients2D.get(y);

        if (!rowGradientMap.containsKey(x)) {
            rowGradientMap.put(x, Vector2D.randomUnitVect(random));
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

            Vector2D position = new Vector2D(xPos, yPos);

            // Bounds
            int x0 = (int) Math.floor(xPos);
            int x1 = x0+1;
            int y0 = (int) Math.floor(yPos);
            int y1 = y0+1;

            // Distances
            Vector2D v0 = Vector2D.subtract(position, new Vector2D(x0, y0));
            Vector2D v1 = Vector2D.subtract(position, new Vector2D(x1, y0));
            Vector2D v2 = Vector2D.subtract(position, new Vector2D(x0, y1));
            Vector2D v3 = Vector2D.subtract(position, new Vector2D(x1, y1));

            // Vertical displacement (delta)
            float d0 = (float) Vector2D.dot(v0, getGradient(x0, y0));
            float d1 = (float) Vector2D.dot(v1, getGradient(x1, y0));
            float d2 = (float) Vector2D.dot(v2, getGradient(x0, y1));
            float d3 = (float) Vector2D.dot(v3, getGradient(x1, y1));

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
