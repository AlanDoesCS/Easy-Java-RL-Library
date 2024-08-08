package Tools;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowEvent;

public abstract class Visualiser extends JFrame {
    final int width, height;
    JPanel panel;

    /**
     * Converts a float value to a grayscale color in the range [min, max]
     *
     * @param v the float value to convert in range [min, max]
     * @param min the minimum value that v can take
     * @param max the maximum value that v can take
     * @return new Color instance of grayscale color
     */
    static Color colorOf(float v, float min, float max) {
        float normalised = math.normalise(v, min, max);
        int b = (int) (255 * (1 - normalised)); // invert brightness (greyscale) - min is white, max is black
        return new Color(b, b, b);
    }

    public void drawCenteredCircle(Graphics2D g, int x, int y, int r) {
        x = x-(r/2);
        y = y-(r/2);
        g.fillOval(x,y,r,r);
    }

    static Color fadeColor(Color startColor, float progress, Color finishColor) {
        int segment_r = (int) math.lerp(progress, startColor.getRed(), finishColor.getRed());
        int segment_g = (int) math.lerp(progress, startColor.getGreen(), finishColor.getGreen());
        int segment_b = (int) math.lerp(progress, startColor.getBlue(), finishColor.getBlue());

        return new Color(segment_r, segment_g, segment_b);
    }

    static Color multiplyColor(Color color, float multiplier) {
        int r = (int) math.clamp((color.getRed() * multiplier), 0f, 255f);
        int g = (int) math.clamp((color.getGreen() * multiplier), 0f, 255f);
        int b = (int) math.clamp((color.getBlue() * multiplier), 0f, 255f);

        return new Color(r, g, b);
    }

    static Color brighter(Color color) {
        return multiplyColor(color, 1.8f);
    }

    static Color darker(Color color) {
        return multiplyColor(color, 0.2f);
    }

    public Visualiser(String title, int width, int height) {
        super(title);
        this.width = width;
        this.height = height;
    }
}
