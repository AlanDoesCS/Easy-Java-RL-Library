package Training.EnvironmentTable;

import Training.Environment;
import Training.PerlinGridEnvironment;

public class PerlinGridEnvTableEntry extends EnvTableEntry {
    final int octaves;
    final float persistence;
    final float step;

    public PerlinGridEnvTableEntry(int octaves, float persistence, float step) {
        this.octaves = octaves;
        this.persistence = persistence;
        this.step = step;
    }

    @Override
    public Environment createInstance(EnvTable table) {
        return new PerlinGridEnvironment(
                table.getGridWidth(), table.getGridHeight(), octaves, persistence, step
        );
    }
}
