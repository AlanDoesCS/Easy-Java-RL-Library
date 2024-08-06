package Tools;

import Structures.Vector2D;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.Arrays;
import java.util.List;

public class GraphPlotter extends JFrame {

    private final List<Vector2D> points;
    private final String graphType;

    private final boolean useEase;  // whether to use an ease function

    public GraphPlotter(String graphTitle, String graphType, String XAxisLabel, String YAxisLabel, List<Vector2D> points, String... varargs) {
        this.graphType = graphType;
        this.points = points;
        List<String> varargs1 = Arrays.asList(varargs);
        this.useEase = varargs1.contains("ease");

        setTitle(graphTitle);
        setSize(600, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        add(new GraphPanel());
    }

    class GraphPanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int padding = 40;
            int width = getWidth() - 2 * padding;
            int height = getHeight() - 2 * padding;

            // draw axes
            g2.drawLine(padding, getHeight() - padding, padding, padding);
            g2.drawLine(padding, getHeight() - padding, getWidth() - padding, getHeight() - padding);

            // plot points
            g2.setColor(Color.RED);
            for (int i = 0; i < points.size(); i++) {
                Vector2D point = points.get(i);
                int x = (int) (point.getI() / 100 * width) + padding;
                int y = getHeight() - ((int) (point.getJ() / 100 * height) + padding);

                if (graphType.equalsIgnoreCase("scatter")) {
                    g2.fillOval(x - 3, y - 3, 6, 6);
                } else if (graphType.equalsIgnoreCase("plot")) {
                    if (i > 0) {
                        Vector2D prevPoint = points.get(i - 1);
                        int prevX = (int) (prevPoint.getI() / 100 * width) + padding;
                        int prevY = getHeight() - ((int) (prevPoint.getJ() / 100 * height) + padding);

                        if (useEase) {
                            drawEased(g2, prevX, prevY, x, y);
                        } else {
                            g2.drawLine(prevX, prevY, x, y);
                        }
                    }
                }
            }
        }

        private void drawEased(Graphics2D g2, int x1, int y1, int x2, int y2) {
            int ctrlX1 = x1 + (x2 - x1) / 3;
            int ctrlX2 = x1 + 2 * (x2 - x1) / 3;

            Path2D path = new Path2D.Double();
            path.moveTo(x1, y1);
            path.curveTo(ctrlX1, y1, ctrlX2, y2, x2, y2);
            g2.draw(path);
        }
    }
}