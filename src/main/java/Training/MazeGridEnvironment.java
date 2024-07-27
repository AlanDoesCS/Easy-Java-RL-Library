package Training;

import Structures.Vector2D;
import Tools.math;

public class MazeGridEnvironment extends GridEnvironment {  // Maze generated using recursive backtracking
    private static final float WALL = 1f;
    private static final float PATH = 0f;

    public MazeGridEnvironment(int width, int height) {
        super(width, height);
        fill();
        setAgentPosition(findValidPositionInBounds());  // represents starting position
        setGoalPosition(findValidPositionInBounds());
    }

    @Override
    void fill() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                set(x, y, 1); // 1 represents a wall
            }
        }

        generateMaze((int) getAgentPosition().getI(), (int) getAgentPosition().getJ());
    }

    private void generateMaze(int x, int y) {
        set(x, y, PATH);

        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        shuffleArray(directions);

        for (int[] dir : directions) {
            int newX = x + dir[0] * 2;
            int newY = y + dir[1] * 2;

            if (isInBounds(newX, newY) && get(newX, newY) == WALL) {
                set(x + dir[0], y + dir[1], PATH);
                generateMaze(newX, newY);
            }
        }
    }

    private Vector2D findValidPositionInBounds() {  // to be used for finding
        Vector2D position;
        do {
            position = getRandomCoordinateInBounds();
            position.set((int) position.getI(), (int) position.getJ());
        } while (get((int) position.getI(), (int) position.getJ()) == WALL);

        return position;
    }

    @Override
    public void randomize() {
        refill();
        setAgentPosition(findValidPositionInBounds());
        setGoalPosition(findValidPositionInBounds());
    }

    private void shuffleArray(int[][] array) {  // Fisher-Yates algorithm
        for (int i = array.length - 1; i > 0; i--) {
            int index = math.randomInt(0, i);
            int[] temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }
}
