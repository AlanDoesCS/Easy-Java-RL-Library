package Structures;

import Tools.math;

import java.util.Random;

public class Vector2 {
    double x, y;

    public Vector2(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public Vector2(Vector2 agentPosition) {
        this.x = agentPosition.x;
        this.y = agentPosition.y;
    }

    public static double dot(Vector2 a, Vector2 b) {
        return a.x *b.x + a.y *b.y;
    }

    public static Vector2 subtract(Vector2 A, Vector2 B) {
        return new Vector2(A.x -B.x, A.y -B.y);
    }

    public static Vector2 randomUnitVect(Random random) {
        Vector2 res = new Vector2(math.randomDouble(-1, 1, random), math.randomDouble(-1, 1, random));
        res.normalise();
        return res;
    }

    public void normalise() {
        double magnitude = Math.sqrt(x * x + y * y);
        x /= magnitude;
        y /= magnitude;
    }
    public static Vector2 normalise(Vector2 vector) {
        double magnitude = Math.sqrt(vector.x * vector.x + vector.y * vector.y);
        return new Vector2(vector.x / magnitude, vector.y / magnitude);
    }
    public static Vector2 normalise(Vector2 vector, double max_x, double max_y) {
        return new Vector2(
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

    public Vector2 copy() {
        return new Vector2(x, y);
    }

    public boolean equals(Vector2 other) {
        return (x == other.x) && (y == other.y);
    }

    public double distanceTo(Vector2 goalPosition) {
        double dx = x - goalPosition.x;
        double dy = y - goalPosition.y;
        return Math.sqrt(dx*dx + dy*dy);
    }

    public double manhattanDistanceTo(Vector2 goalPosition) {
        return Math.abs(x - goalPosition.x) + Math.abs(y - goalPosition.y);
    }
}
