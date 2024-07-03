package Tools.Pathfinding;

import Training.GridEnvironment;

public class NodeGrid {
    private GraphNode[][] nodes;
    private final int width, height;

    public NodeGrid(int width, int height, GridEnvironment environment) {
        this.width = width;
        this.height = height;
        nodes = new GraphNode[width][height];
        initializeNodes(environment);
    }

    private void initializeNodes(GridEnvironment environment) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                nodes[x][y] = new GraphNode(x, y, environment.get(x, y));
            }
        }
    }

    public GraphNode getNode(int x, int y) {
        return nodes[x][y];
    }

    public void setNode(int x, int y, GraphNode node) {
        nodes[x][y] = node;
    }
}

