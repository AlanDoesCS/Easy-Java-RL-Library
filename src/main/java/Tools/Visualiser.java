package Tools;

import javax.swing.*;
import java.awt.*;

public abstract class Visualiser extends JFrame {
    final int width, height;
    JPanel panel;

    static Color colorOf(float noise) { // 1: Black, -1: White, (continuous)
        float normNoise = (noise + 1) / 2;
        normNoise = Math.min(Math.max(normNoise, 0), 1);
        int b = (int) (255 * (1 - normNoise)); // invert brightness (greyscale)
        return new Color(b, b, b);
    }

    public void drawCenteredCircle(Graphics2D g, int x, int y, int r) {
        x = x-(r/2);
        y = y-(r/2);
        g.fillOval(x,y,r,r);
    }

    static Color fadeColor(Color startColor, float progress, Color finishColor) {
        int segment_r = (int) (startColor.getRed() + progress * (finishColor.getRed() - startColor.getRed()));
        int segment_g = (int) (startColor.getGreen() + progress * (finishColor.getGreen() - startColor.getGreen()));
        int segment_b = (int) (startColor.getBlue() + progress * (finishColor.getBlue() - startColor.getBlue()));

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
