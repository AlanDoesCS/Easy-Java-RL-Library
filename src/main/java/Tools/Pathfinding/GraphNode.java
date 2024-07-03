package Tools.Pathfinding;

public class GraphNode {
    int x, y;
    float weight;
    GraphNode previous = null;
    float dist = Float.MAX_VALUE; // Infinity

    public GraphNode(int x, int y, float weight) {
        this.x = x;
        this.y = y;
        this.weight = weight;
    }

    public int compareTo(GraphNode o) {
        return Float.compare(this.dist, o.dist);
    }

    public void setPrevious(GraphNode previous) {
        this.previous = previous;
    }

    @Override
    public String toString() {
        return "(" + x + ", " + y + ", " + weight + ")";
    }
}
