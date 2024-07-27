package Structures;

import Tools.math;

import java.util.Random;

public class Vector2D {
    float i, j;

    public Vector2D(float i, float j) {
        this.i = i;
        this.j = j;
    }

    public static float dot(Vector2D a, Vector2D b) {
        return a.i*b.i + a.j*b.j;
    }

    public static Vector2D subtract(Vector2D A, Vector2D B) {
        return new Vector2D(A.i-B.i, A.j-B.j);
    }

    public static Vector2D randomUnitVect(Random random) {
        Vector2D res = new Vector2D(math.randomFloat(-1, 1, random), math.randomFloat(-1, 1, random));
        res.normalise();
        return res;
    }

    public void normalise() {
        float magnitude = math.fastSqrt(i*i+j*j);
        i /= magnitude;
        j /= magnitude;
    }

    public float getI() {
        return i;
    }
    public float getJ() {
        return j;
    }

    public String toString() {
        return "Vector(" + "i=" + i + ", j=" + j + ')';
    }

    public void add(float I, float J) {
        i += I;
        j += J;
    }

    public void multiplyI(float multiplier) {
        this.i *= multiplier;
    }
    public void multiplyJ(float multiplier) {
        this.j *= multiplier;
    }

    public void set(int i, int j) {
        this.i = i;
        this.j = j;
    }

    public void set(float i, float j) {
        this.i = i;
        this.j = j;
    }
}
