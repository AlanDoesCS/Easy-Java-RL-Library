package Training;

import Structures.Vector2D;
import Tools.math;

import java.util.Stack;

public class MazeGridEnvironment extends GridEnvironment {  // Maze generated using a modified recursive backtracking approach that uses a stack instead
    private static final float WALL = 1f;
    private static final float PATH = -1f;

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
                set(x, y, WALL); // 1 represents a wall
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
            int x = (int) current.getI();
            int y = (int) current.getJ();

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

    boolean IsValidPositionInBounds(int x, int y) {
        return get(x, y) == PATH;
    }
    private Vector2D findValidPositionInBounds() {  // to be used for finding
        Vector2D position;
        do {
            position = getRandomCoordinateInBounds();
            position.set((int) position.getI(), (int) position.getJ());
        } while (!IsValidPositionInBounds((int) position.getI(), (int) position.getJ()));

        System.out.println("Found valid position: " + position + " has: " + get((int) position.getI(), (int) position.getJ()));

        return position;
    }

    @Override
    public void randomize() {
        refill();
        setAgentPosition(findValidPositionInBounds());
        setGoalPosition(findValidPositionInBounds());
    }

    @Override
    public MoveResult step(int action) {
        float reward = 0;
        Vector2D currentPosition = getAgentPosition();
        Vector2D newPosition = currentPosition.copy();

        PerlinGridEnvironment.getNewPosFromAction(action, newPosition);

        if (IsValidPositionInBounds((int) newPosition.getI(), (int) newPosition.getJ())) {
            setAgentPosition(newPosition);
        }

        boolean done = newPosition.equals(getGoalPosition());

        reward += done ? getCompletionReward() : -1;

        return new MoveResult(getState(), reward, done);
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
