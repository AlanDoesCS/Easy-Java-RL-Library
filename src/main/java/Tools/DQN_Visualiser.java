package Tools;

import javax.swing.*;

import java.awt.*;
import java.util.ArrayList;

import Structures.DQN;
import Structures.Layer;

public class DQN_Visualiser extends JFrame {
    static final int width=1280, height=900;

    public DQN_Visualiser(DQN network) {
        super("Deep Q Network Visualiser");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(width, height);

        // Node visualisation
        int numLayers = network.numLayers();
        float hSpacing = (float) width /(numLayers+1);

        ArrayList<Node[]> layers = new ArrayList<>(numLayers);

        layers.add(Node.ArrayOfInput(network.getInputSize(), height, hSpacing)); // input

        for (int layerIndex=1; layerIndex<numLayers-1; layerIndex++) { // hidden
            Layer layer = network.getLayer(layerIndex-1);
            layers.add(Node.ArrayOf(layer, height, hSpacing, layerIndex));
        }

        layers.add(Node.ArrayOf(network.getOutputLayer(), height, hSpacing, numLayers-1)); // output


        JPanel pn = new JPanel() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2=(Graphics2D)g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setStroke(new BasicStroke(2));

                final int lastIndex = numLayers - 1;
                int depth = 0;
                for (Node[] nodeLayer : layers) {
                    for (Node n : nodeLayer) {
                        Color col;

                        if (depth == 0) { // INPUT LAYER
                            col = Node.input_Color;
                        } else if (depth == lastIndex) { // OUTPUT LAYER
                            col = Node.output_Color;
                        } else { // HIDDEN LAYER
                            col = Node.hidden_Color;
                        }

                        g2.setColor(col);

                        int x = n.x-(n.size_x/2);
                        int y = n.y-(n.size_y/2);

                        g2.fillOval(x, y, n.size_x, n.size_y);

                        if (depth != 0) {
                            for (Node other : layers.get(depth-1)) {
                                g2.setColor(Color.BLACK);
                                // TODO: PROPER NODE WEIGHTS: Color Blue and red + weight strength
                                g2.drawLine(x, y, other.x, other.y);
                            }
                        }
                    }

                    depth++;
                }
            }
        };

        add(pn, BorderLayout.CENTER);
        pn.setLayout(null);

        setVisible(true);
    }
}
