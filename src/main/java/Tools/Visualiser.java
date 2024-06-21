package Tools;

import Structures.Vector2D;

import javax.swing.*;
import java.awt.*;

public abstract class Visualiser extends JFrame {
    final int width, height;
    JPanel panel;

    static Color colorOf(float noise) { // -1: Black, 1: White, (continuous)
        float normNoise = (noise+1)/2;
        normNoise = Math.min(Math.max(normNoise, 0), 1);
        int b = (int) (255*normNoise); // brightness (greyscale)
        return new Color(b, b, b);
    }

    public void drawCenteredCircle(Graphics2D g, int x, int y, int r) {
        x = x-(r/2);
        y = y-(r/2);
        g.fillOval(x,y,r,r);
    }

    public Visualiser(String title, int width, int height) {
        super(title);
        this.width = width;
        this.height = height;
    }
}
