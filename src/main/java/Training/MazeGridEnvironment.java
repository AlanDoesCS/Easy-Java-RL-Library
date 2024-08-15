package Training;

import Structures.Vector2D;
import Tools.math;

import java.util.Stack;

public class MazeGridEnvironment extends GridEnvironment {  // Maze generated using a modified recursive backtracking approach that uses a stack instead
    private static final float WALL = 100;
    private static final float PATH = 0.1f;

    public MazeGridEnvironment(int width, int height) {
        super(width, height);
        fill();
        setAgentPosition(findValidPositionInBounds());  // represents starting position
        setStartPosition(getAgentPosition());
        setGoalPosition(findValidPositionInBounds());
    }

    @Override
    void fill() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                set(x, y, WALL);
            }
        }

        int startX = math.randomInt(0, width - 1);
        int startY = math.randomInt(0, height - 1);

        generateMaze(startX, startY);
    }

    private void generateMaze(int startX, int startY) {
        // keep track of cells to visit: prevent reaching max recursive depth
        Stack<Vector2D> stack = new Stack<>();
        stack.push(new Vector2D(startX, startY));
        set(startX, startY, PATH);

        while (!stack.isEmpty()) {  // visit all accessible cells
            Vector2D current = stack.peek();
            int x = (int) current.getX();
            int y = (int) current.getY();

            int[][] directions = {{0, 2}, {2, 0}, {0, -2}, {-2, 0}};

            // randomize direction order to ensure maze randomness
            shuffleArray(directions);

            boolean moved = false;

            for (int[] dir : directions) {
                int newX = x + dir[0];
                int newY = y + dir[1];

                if (isInBounds(newX, newY) && get(newX, newY) == WALL) {
                    set(x + dir[0] / 2, y + dir[1] / 2, PATH);
                    set(newX, newY, PATH);

                    stack.push(new Vector2D(newX, newY));
                    moved = true;
                    break;
                }
            }

            // if we did not move to a new cell, backtrack
            if (!moved) {
                stack.pop();
            }
        }
    }

    @Override
    boolean isValidPositionInBounds(int x, int y) {
        if (x < 0 || x >= getWidth() || y < 0 || y >= getHeight()) return false;
        return get(x, y) == PATH;
    }

    public Vector2D findValidPositionInBounds() {  // to be used for finding
        Vector2D position;
        do {
            position = getRandomCoordinateInBounds();
            position.set((int) position.getX(), (int) position.getY());
        } while (!isValidPositionInBounds((int) position.getX(), (int) position.getY()));

        return position;
    }

    @Override
    public void randomize() {
        refill();
        setAgentPosition(findValidPositionInBounds());
        setStartPosition(getAgentPosition());
        setGoalPosition(findValidPositionInBounds());
        this.currentSteps = 0;
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
