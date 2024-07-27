package Training.EnvironmentTable;

import Training.Environment;
import Training.RandomGridEnvironment;

public class RandomGridEnvTableEntry extends EnvTableEntry {
    final int octaves;
    final float persistence;
    final float step;

    public RandomGridEnvTableEntry(int octaves, float persistence, float step) {
        this.octaves = octaves;
        this.persistence = persistence;
        this.step = step;
    }

    @Override
    public Environment createInstance(EnvTable table) {
        return new RandomGridEnvironment(
                table.getGridWidth(), table.getGridHeight()
        );
    }
}
