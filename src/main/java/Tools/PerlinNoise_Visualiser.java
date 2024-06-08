package Tools;

import javax.swing.*;

import java.awt.*;

import Structures.Matrix;

public class PerlinNoise_Visualiser extends JFrame {
    static final int width=1280, height=900;

    private Color colorOf(float noise) { // noise is in range [-1, 1]
        float normNoise = (noise+1)/2;
        normNoise = Math.min(Math.max(normNoise, 0), 1);
        int b = (int) (255*normNoise); // brightness (greyscale)
        return new Color(b, b, b);
    }

    public PerlinNoise_Visualiser(Perlin1D perlin, float step) {
        super("1D Perlin Noise Visualiser");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        int xPadding = 5; // px
        int yPadding = 5; // px
        int vRange = height - 2*yPadding;
        int midHeight = vRange / 2 + yPadding;

        JPanel pn = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2=(Graphics2D)g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.black);

                for (int x=0; x<width-xPadding; x++) {
                    float noise = perlin.noise(x*step);
                    g2.setColor(colorOf(noise));
                    int vPos = (int) (midHeight + noise*(vRange/2));
                    int hPos = xPadding+x;
                    g2.drawRect(hPos, vPos, 1, 1);
                }
            }
        };

        add(pn);
        setVisible(true);
    }

    public PerlinNoise_Visualiser(Perlin2D perlin, float step) {
        super("2D Perlin Noise Visualiser");
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
                        g2.setColor(colorOf(noise));
                        g2.drawRect(x, y, 1, 1);
                    }
                }
            }
        };

        add(pn);
        setVisible(true);
    }
}

