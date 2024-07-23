package Tools;

import Structures.Vector2D;
import Training.PerlinGridEnvironment;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class Environment_Visualiser extends Visualiser {
    static final int width = 720, height = 720, pointRadius = 5;
    final int squareWIDTH, squareHEIGHT;
    ArrayList<ArrayList<Vector2D>> pathsFollowed = new ArrayList<>();
    ArrayList<Color> pathColors = new ArrayList<>();

    public Environment_Visualiser(PerlinGridEnvironment environment) {
        super("Perlin Grid environment", width, height);

        if (environment == null) throw new NullPointerException("Cannot visualise an environment if it is null!");

        final int envWIDTH = environment.getWidth();
        final int envHEIGHT = environment.getHeight();
        this.squareWIDTH = Math.max(width / envWIDTH, 1);
        this.squareHEIGHT = Math.max(height / envHEIGHT, 1);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        this.panel = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.black);

                // Create grid squares
                for (int y=0; y<envHEIGHT; y++) {
                    for (int x=0; x<envWIDTH; x++) {
                        g2.setColor(colorOf(environment.get(x, y)));
                        g2.fillRect(x*squareWIDTH, y*squareHEIGHT, squareWIDTH, squareHEIGHT);
                    }
                }

                // Draw paths
                for (int pathIndex=0; pathIndex<pathsFollowed.size(); pathIndex++) {
                    ArrayList<Vector2D> path = pathsFollowed.get(pathIndex);
                    Color color = pathColors.get(pathIndex);
                    g2.setColor(color);


                    for (int i=0; i<path.size()-1; i++) {
                        Vector2D c = path.get(i);
                        Vector2D n = path.get(i+1);
                        g2.drawLine(
                                (int) ((c.getI()+0.5)*squareWIDTH), (int) ((c.getJ()+0.5)*squareHEIGHT),
                                (int) ((n.getI()+0.5)*squareWIDTH), (int) ((n.getJ()+0.5)*squareHEIGHT)
                        );
                    }
                }
            }
        };

        add(panel);
        setVisible(true);
    }

    public void addPath(ArrayList<Vector2D> path, Color color) {
        pathsFollowed.add(path);
        pathColors.add(color);
    }
}