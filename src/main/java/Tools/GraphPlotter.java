package Tools;

import Structures.Vector2D;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.*;
import java.util.*;
import java.util.List;

public class GraphPlotter extends JFrame {
    // STATIC VARS
    public static enum Types {
        SCATTER, LINE
    }

    // INSTANCE VARS
    private PriorityQueue<Vector2D> points = new PriorityQueue<>(Comparator.comparing(Vector2D::getI));
    private Types graphType;
    private String XAxisLabel, YAxisLabel;
    List<String> args;

    private boolean useEase;  // whether to use an ease function

    private GraphPanel graphPanel;

    //For calculating getting grid bounds:
    private float minX=Float.MAX_VALUE, maxX=Float.MIN_VALUE, minY=Float.MAX_VALUE, maxY=Float.MIN_VALUE;

    //Styling:
    private int padding = 40;

    private void init(String graphTitle, Types graphType, String XAxisLabel, String YAxisLabel, List<Vector2D> points, String... varargs) {
        this.graphType = graphType;
        this.points.addAll(points);
        this.args = Arrays.asList(varargs);
        this.useEase = this.args.contains("ease");

        this.XAxisLabel = XAxisLabel;
        this.YAxisLabel = YAxisLabel;

        setTitle(graphTitle);
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setLocationRelativeTo(null);

        add(new GraphPanel());
    }
    public GraphPlotter(String graphTitle, Types graphType, String XAxisLabel, String YAxisLabel, List<Vector2D> points, String... varargs) {
        init(graphTitle, graphType, XAxisLabel, YAxisLabel, points, varargs);
    }
    public GraphPlotter(String graphTitle, Types graphType, String XAxisLabel, String YAxisLabel, String... varargs) {
        init(graphTitle, graphType, XAxisLabel, YAxisLabel, new ArrayList<>(), varargs);
    }

    class GraphPanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth() - 2 * padding;
            int height = getHeight() - 2 * padding;

            // draw axes
            g2.drawLine(padding, getHeight() - padding, padding, padding);
            g2.drawLine(padding, getHeight() - padding, getWidth() - padding, getHeight() - padding);

            // plot points
            g2.setColor(Color.RED);

            PriorityQueue<Vector2D> temp = new PriorityQueue<>(points);
            Vector2D prevPoint = temp.peek();
            int i=0;
            while (!temp.isEmpty()) {
                Vector2D point = temp.poll();
                int x = (int) (point.getI() / 100 * width) + padding;
                int y = getHeight() - ((int) (point.getJ() / 100 * height) + padding);

                switch (graphType) {
                    case SCATTER:
                        drawScatterPoint(g2, x, y);
                        break;
                    case LINE:
                        drawPlotPoint(g2, prevPoint, x, y, width, height, padding, useEase);
                        break;
                    default:
                        drawScatterPoint(g2, x, y);
                }
                prevPoint = point;
            }
        }

        private void drawScatterPoint(Graphics2D g2, int x, int y) {
            g2.fillOval(x - 3, y - 3, 6, 6);
        }

        private void drawPlotPoint(Graphics2D g2, Vector2D prevPoint, int x, int y, int width, int height, int padding, boolean useEase) {
            int prevX = (int) (prevPoint.getI() / 100 * width) + padding;
            int prevY = getHeight() - ((int) (prevPoint.getJ() / 100 * height) + padding);

            if (useEase) {
                drawEased(g2, prevX, prevY, x, y);
            } else {
                g2.drawLine(prevX, prevY, x, y);
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

    public void addPoint(Vector2D point) {
        points.add(point);

        // update graph bounds:
        final float x = point.getI();
        final float y = point.getJ();

        if (x < minX) {
            minX = x;
        }
        if (x > maxX) {
            maxX = x;
        }
        if (y < minY) {
            minY = y;
        }
        if (y > maxY) {
            maxY = y;
        }
    }

    public void reset() {
        //reset data
        points.clear();
        minX = Float.MAX_VALUE;
        maxX = Float.MIN_VALUE;
        minY = Float.MAX_VALUE;
        maxY = Float.MIN_VALUE;
        repaint();
    }

    public void plot() {
        SwingUtilities.invokeLater(this::repaint);
    }

    public int getNumPoints() {
        return points.size();
    }
}