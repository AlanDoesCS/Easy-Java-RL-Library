package Tools.Pathfinding;

import Structures.Vector2D;
import Training.GridEnvironment;

import java.util.*;

public class Pathfinder {
    private static final Comparator<GraphNode> nodeComparator = new Comparator<>() {
        @Override
        public int compare(final GraphNode n1, final GraphNode n2) {
            return n1.compareTo(n2);
        }
    };

    public static ArrayList<Vector2D> dijkstra(Vector2D start, Vector2D end, GridEnvironment environment) {
        ArrayList<Vector2D> path = new ArrayList<>();
        NodeGrid nodeGrid = new NodeGrid(environment.width, environment.height, environment);
        PriorityQueue<GraphNode> Q = new PriorityQueue<>(environment.getNumSquares(), nodeComparator);

        int x0 = (int) start.getX();
        int y0 = (int) start.getY();
        int x1 = (int) end.getX();
        int y1 = (int) end.getY();

        GraphNode startNode = nodeGrid.getNode(x0, y0);
        startNode.dist = 0;
        Q.add(startNode);

        while (!Q.isEmpty()) {
            GraphNode current = Q.poll();

            if (current.x == x1 && current.y == y1) {
                while (current != null) {
                    path.add(new Vector2D(current.x, current.y));
                    current = current.previous;
                }
                Collections.reverse(path);
                return path;
            }

            for (int i = 0; i < 4; i++) {
                int newX = current.x + dx[i];
                int newY = current.y + dy[i];
                if (isValid(newX, newY, environment.width, environment.height)) {
                    GraphNode neighbor = nodeGrid.getNode(newX, newY);
                    if (neighbor != null) {
                        float newDist = current.dist + (neighbor.weight+1);
                        if (newDist < neighbor.dist) {
                            Q.remove(neighbor); // Remove and reinsert to update the priority queue
                            neighbor.dist = newDist;
                            neighbor.previous = current;
                            Q.add(neighbor);
                        }
                    }
                }
            }
        }

        return path; // No path found
    }

    private static boolean isValid(int x, int y, int width, int height) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    private static final int[] dx = {-1, 1, 0, 0};
    private static final int[] dy = {0, 0, -1, 1};
}