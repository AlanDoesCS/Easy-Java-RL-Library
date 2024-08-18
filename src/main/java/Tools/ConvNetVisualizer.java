package Tools;

import Structures.ConvLayer;
import Structures.DQN;
import Structures.Layer;

import javax.swing.*;
import java.awt.*;

public class ConvNetVisualizer extends JFrame {
    private final DQN network;
    private final int padding = 50;
    private final int layerSpacing = 200;
    private final int neuronSize = 30;
    private final int scale = 5;

    public ConvNetVisualizer(DQN network) {
        this.network = network;
        setTitle("Convolutional Neural Network Visualizer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setLocationRelativeTo(null);
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int x = padding;
        int y = getHeight() / 2;

        // Draw input layer
        drawLayer(g2d, x, y, network.getInputSize(), 3, "Input");
        x += layerSpacing;

        // Draw convolutional layers
        for (int i = 0; i < network.numLayers(); i++) {
            Layer layer = network.getLayer(i);
            if (layer instanceof ConvLayer) {
                ConvLayer convLayer = (ConvLayer) layer;
                drawConvLayer(g2d, x, y, convLayer);
                x += layerSpacing;
            }
        }
    }

    private void drawLayer(Graphics2D g2d, int x, int y, int size, int depth, String label) {
        int width = (int) Math.sqrt(size / depth);
        int height = size / (width * depth);

        g2d.setColor(Color.BLUE);
        g2d.fillRect(x, y - (height * scale) / 2, width * scale, height * scale);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y - (height * scale) / 2, width * scale, height * scale);

        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString(label, x, y + (height * scale) / 2 + 20);
        g2d.drawString(width + "x" + height + "x" + depth, x, y + (height * scale) / 2 + 35);
    }

    private void drawConvLayer(Graphics2D g2d, int x, int y, ConvLayer layer) {
        int outputWidth = layer.getOutputWidth();
        int outputHeight = layer.getOutputHeight();
        int outputDepth = layer.getOutputDepth();

        g2d.setColor(Color.RED);
        g2d.fillRect(x, y - (outputHeight * scale) / 2, outputWidth * scale, outputHeight * scale);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y - (outputHeight * scale) / 2, outputWidth * scale, outputHeight * scale);

        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString("Conv", x, y + (outputHeight * scale) / 2 + 20);
        g2d.drawString(outputWidth + "x" + outputHeight + "x" + outputDepth, x, y + (outputHeight * scale) / 2 + 35);
        g2d.drawString(layer.filterSize + "x" + layer.filterSize + " filters", x, y + (outputHeight * scale) / 2 + 50);
    }

    public static void visualize(DQN network) {
        SwingUtilities.invokeLater(() -> {
            ConvNetVisualizer visualizer = new ConvNetVisualizer(network);
            visualizer.setVisible(true);
        });
    }
}
