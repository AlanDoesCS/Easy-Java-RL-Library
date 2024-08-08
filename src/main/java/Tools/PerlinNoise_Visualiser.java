package Tools;

import javax.swing.*;

import java.awt.*;

import Structures.Matrix;

public class PerlinNoise_Visualiser extends Visualiser {

    public PerlinNoise_Visualiser(Perlin1D perlin, float step) {
        super("1D Perlin Noise Visualiser", 1000, 1000);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        int xPadding = 5; // px
        int yPadding = 5; // px
        int vRange = height - 2*yPadding;
        int midHeight = vRange / 2 + yPadding;

        this.panel = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2=(Graphics2D)g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.black);

                for (int x=0; x<width-xPadding; x++) {
                    float noise = perlin.noise(x*step);
                    g2.setColor(colorOf(noise, -1, 1));
                    int vPos = (int) (midHeight + noise*(vRange/2));
                    int hPos = xPadding+x;
                    g2.drawRect(hPos, vPos, 1, 1);
                }
            }
        };

        add(panel);
        setVisible(true);
    }

    public PerlinNoise_Visualiser(Perlin2D perlin, float step) {
        super("2D Perlin Noise Visualiser", 1000, 1000);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        Matrix pxs = perlin.toMatrix(width, height, step);

        JPanel pn = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2=(Graphics2D)g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.black);

                for (int y=0; y<height; y++) {
                    for (int x=0; x<width; x++) {
                        float noise = pxs.get(x, y);
                        g2.setColor(colorOf(noise, -1, 1));
                        g2.drawRect(x, y, 1, 1);
                    }
                }
            }
        };

        add(pn);
        setVisible(true);
    }
}

