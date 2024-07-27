package Training.EnvironmentTable;

import Training.Environment;

public abstract class EnvTableEntry {
    public abstract Environment createInstance(EnvTable table);
}
