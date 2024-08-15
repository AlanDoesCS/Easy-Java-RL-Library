package Structures;

import Tools.math;

import java.util.Random;

public class Vector2D {
    float x, y;

    public Vector2D(float x, float y) {
        this.x = x;
        this.y = y;
    }

    public Vector2D(Vector2D agentPosition) {
        this.x = agentPosition.x;
        this.y = agentPosition.y;
    }

    public static float dot(Vector2D a, Vector2D b) {
        return a.x *b.x + a.y *b.y;
    }

    public static Vector2D subtract(Vector2D A, Vector2D B) {
        return new Vector2D(A.x -B.x, A.y -B.y);
    }

    public static Vector2D randomUnitVect(Random random) {
        Vector2D res = new Vector2D(math.randomFloat(-1, 1, random), math.randomFloat(-1, 1, random));
        res.normalise();
        return res;
    }

    public void normalise() {
        float magnitude = math.fastSqrt(x * x + y * y);
        x /= magnitude;
        y /= magnitude;
    }

    public float getX() {
        return x;
    }
    public float getY() {
        return y;
    }

    public String toString() {
        return "(" + "x=" + x + ", y=" + y + ')';
    }

    public void add(float I, float J) {
        x += I;
        y += J;
    }

    public void multiplyX(float multiplier) {
        this.x *= multiplier;
    }
    public void multiplyY(float multiplier) {
        this.y *= multiplier;
    }

    public void set(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void set(float x, float y) {
        this.x = x;
        this.y = y;
    }

    public void addI(float amount) {
        this.x += amount;
    }
    public void addJ(float amount) {
        this.y += amount;
    }

    public Vector2D copy() {
        return new Vector2D(x, y);
    }

    public boolean equals(Vector2D other) {
        return (x == other.x) && (y == other.y);
    }
}
