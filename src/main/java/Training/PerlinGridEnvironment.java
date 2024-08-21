package Training;

import Structures.MatrixDouble;
import Tools.Perlin2D;

public class PerlinGridEnvironment extends GridEnvironment {
    float step, persistence;
    int octaves;
    Perlin2D perlin;
    MatrixDouble gridAsMatrix;

    public PerlinGridEnvironment(int width, int height, int octaves, float persistence, float step) {
        super(width, height);

        this.perlin = new Perlin2D(octaves, persistence);
        this.step = step;
        this.octaves = octaves;
        this.persistence = persistence;

        fill();
    }

    @Override
    void fill() {
        gridAsMatrix = fill(new MatrixDouble(height, width));
    }

    MatrixDouble fill(MatrixDouble destination) {
        if (destination == null) throw new NullPointerException("MatrixDouble destination is null!");

        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                float noiseValue = perlin.noise(x*step, y*step, 0, 1);
                set(x, y, noiseValue);
                destination.set(x, y, noiseValue);
            }
        }
        return destination;
    }

    @Override
    public void refill() {
        this.perlin = new Perlin2D(octaves, persistence);
        fill();
    }
}
