package Structures;

import Tools.math;

import java.util.Random;

public class Vector2D {
    double x, y;

    public Vector2D(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Vector2D(Vector2D agentPosition) {
        this.x = agentPosition.x;
        this.y = agentPosition.y;
    }

    public static double dot(Vector2D a, Vector2D b) {
        return a.x *b.x + a.y *b.y;
    }

    public static Vector2D subtract(Vector2D A, Vector2D B) {
        return new Vector2D(A.x -B.x, A.y -B.y);
    }

    public static Vector2D randomUnitVect(Random random) {
        Vector2D res = new Vector2D(math.randomDouble(-1, 1, random), math.randomDouble(-1, 1, random));
        res.normalise();
        return res;
    }

    public void normalise() {
        double magnitude = Math.sqrt(x * x + y * y);
        x /= magnitude;
        y /= magnitude;
    }
    public static Vector2D normalise(Vector2D vector) {
        double magnitude = Math.sqrt(vector.x * vector.x + vector.y * vector.y);
        return new Vector2D(vector.x / magnitude, vector.y / magnitude);
    }
    public static Vector2D normalise(Vector2D vector, double max_x, double max_y) {
        return new Vector2D(
                math.normalise(vector.x, 0, max_x),
                math.normalise(vector.y, 0, max_y)
        );
    }

    public double getX() {
        return x;
    }
    public double getY() {
        return y;
    }

    public String toString() {
        return "(x=" + x + ", y=" + y + ')';
    }

    public void add(double I, double J) {
        x += I;
        y += J;
    }

    public void multiplyX(double multiplier) {
        this.x *= multiplier;
    }
    public void multiplyY(double multiplier) {
        this.y *= multiplier;
    }

    public void set(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void set(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public void addX(double amount) {
        this.x += amount;
    }
    public void addY(double amount) {
        this.y += amount;
    }

    public Vector2D copy() {
        return new Vector2D(x, y);
    }

    public boolean equals(Vector2D other) {
        return (x == other.x) && (y == other.y);
    }

    public double distanceTo(Vector2D goalPosition) {
        double dx = x - goalPosition.x;
        double dy = y - goalPosition.y;
        return Math.sqrt(dx*dx + dy*dy);
    }

    public double manhattanDistanceTo(Vector2D goalPosition) {
        return Math.abs(x - goalPosition.x) + Math.abs(y - goalPosition.y);
    }
}
