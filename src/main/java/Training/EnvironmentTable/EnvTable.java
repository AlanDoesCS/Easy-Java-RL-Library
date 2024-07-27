package Training.EnvironmentTable;

import java.util.ArrayList;
import java.util.List;

public class EnvTable {
    final int gridWidth, gridHeight;
    List<EnvTableEntry> entries = new ArrayList<EnvTableEntry>();

    public EnvTable(int gridWidth, int gridHeight) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
    }

    public void addEntry(EnvTableEntry entry) {
        entries.add(entry);
    }

    public void addAllEntries(List<EnvTableEntry> entries) {
        this.entries.addAll(entries);
    }

    public int getGridWidth() {
        return gridWidth;
    }

    public int getGridHeight() {
        return gridHeight;
    }
}
