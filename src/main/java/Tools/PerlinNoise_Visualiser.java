package Tools;

import javax.swing.*;

import java.awt.*;

import Structures.DQN;
import Structures.Matrix;

public class PerlinNoise_Visualiser extends JFrame {
    static final int width=1280, height=900;

    private Color colorOf(float noise) { // noise is in range [-1, 1]
        float normNoise = (noise+1)/2;
        normNoise = Math.min(Math.max(normNoise, 0), 1);
        int b = (int) (255*normNoise); // brightness (greyscale)
        return new Color(b, b, b);
    }

    public PerlinNoise_Visualiser(int dimensions, int octaves, float persistence) {
        super(dimensions+"D Perlin Noise Visualiser");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        PerlinNoise perlin = new PerlinNoise();

        int xPadding = 5; // px
        int yPadding = 5; // px
        int vRange = height - 2*yPadding;
        int midHeight = vRange / 2 + yPadding;

        JPanel pn;

        if (dimensions == 1) { // 1D
            pn = new JPanel() {
                @Override
                public void paint(Graphics g) {
                    Graphics2D g2=(Graphics2D)g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g2.setColor(Color.black);

                    for (int x=0; x<width-xPadding; x++) {
                        float noise = perlin.noise(x*0.001f, octaves, persistence);
                        g2.setColor(colorOf(noise));
                        int vPos = (int) (midHeight + noise*(vRange/2));
                        int hPos = xPadding+x;
                        g2.drawRect(hPos, vPos, 1, 1);
                    }
                }
            };
        } else { // 2D
            Matrix pxs = perlin.toMatrix(width, height, 0.001f, octaves, persistence);

            pn = new JPanel() {
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
        }

        add(pn);
        setVisible(true);
    }
}

