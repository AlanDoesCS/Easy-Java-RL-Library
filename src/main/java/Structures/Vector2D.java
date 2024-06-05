package Structures;

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

    public float getI() {
        return i;
    }

    public float getJ() {
        return j;
    }
}
