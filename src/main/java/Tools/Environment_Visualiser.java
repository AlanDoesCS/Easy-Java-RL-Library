package Tools;

import Structures.Vector2D;
import Training.GridEnvironment;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class Environment_Visualiser extends Visualiser {
    static final int width = 720, height = 720, pointRadius = 5;
    final int squareWIDTH, squareHEIGHT;
    ArrayList<ArrayList<Vector2D>> pathsFollowed = new ArrayList<>();
    ArrayList<Color> pathColors = new ArrayList<>();

    public Environment_Visualiser(GridEnvironment environment) {
        super("Grid environment", width, height);

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
                        g2.setColor(colorOf(environment.get(x, y), 0, 1));
                        g2.fillRect(x*squareWIDTH, y*squareHEIGHT, squareWIDTH, squareHEIGHT);
                    }
                }

                g2.setStroke(new BasicStroke(Math.max(squareWIDTH / 6, 1)));

                // Draw paths
                for (int pathIndex=0; pathIndex<pathsFollowed.size(); pathIndex++) {
                    ArrayList<Vector2D> path = pathsFollowed.get(pathIndex);

                    Color color = pathColors.get(pathIndex);
                    Color startColor = darker(color);
                    Color finishColor = brighter(color);
                    g2.setColor(startColor);

                    for (int i=0; i<path.size(); i++) {
                        Vector2D c = path.get(i);

                        // Calculate the progress along the path (0 to 1)
                        float progress = (float) i / (path.size() - 2);

                        // Interpolate between start and end colors
                        Color segmentColor = fadeColor(startColor, progress, finishColor);
                        g2.setColor(segmentColor);

                        g2.fillRect(
                                (int) (c.getI()*squareWIDTH), (int) (c.getJ()*squareHEIGHT), squareWIDTH, squareHEIGHT
                        );
                    }

                    // Draw the start square as green
                    Vector2D start = path.getFirst();
                    g2.setColor(Color.GREEN);
                    g2.fillRect(
                            (int) (start.getI()*squareWIDTH), (int) (start.getJ()*squareHEIGHT), squareWIDTH, squareHEIGHT
                    );

                    // Draw the finish square as red
                    Vector2D finish = path.getLast();
                    g2.setColor(Color.RED);
                    g2.fillRect(
                            (int) (finish.getI()*squareWIDTH), (int) (finish.getJ()*squareHEIGHT), squareWIDTH, squareHEIGHT
                    );
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
