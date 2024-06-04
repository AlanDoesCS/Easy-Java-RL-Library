package Tools.VisualiserDependencies;

import Structures.Layer;
import Tools.string;

import java.awt.Color;

public class Node {
    public static final Color input_Color = Color.BLUE;
    public static final Color hidden_Color = Color.GRAY;
    public static final Color output_Color = Color.GREEN;

    public String id;
    public final int x, y, size_x, size_y;

    public Node(String n_id, int x_coordinate, int y_coordinate) {
        id = n_id;
        x = x_coordinate;
        y = y_coordinate;
        size_x = 10;
        size_y = 10;
    }

    public static Node[] ArrayOf(Layer layer, int windowHeight, float hSpacing, int layerIndex) {
        int layerSize = layer.getOutputSize();
        return getNodes(layerSize, windowHeight, hSpacing, layerIndex);
    }

    public static Node[] ArrayOfInput(int layerSize, int windowHeight, float hSpacing) {
        return getNodes(layerSize, windowHeight, hSpacing, 0);
    }

    private static Node[] getNodes(int layerSize, float windowHeight, float hSpacing, int layerIndex) {
        Node[] nodesForLayer = new Node[layerSize];

        float vSpacing = windowHeight /(layerSize+1);

        for (int nodeIndex=0; nodeIndex<layerSize; nodeIndex++) {
            //Node Name
            String Node_ID = string.intToAlphabet(layerIndex+1) + nodeIndex;
            int x_coordinate = (int) (hSpacing + layerIndex*hSpacing);
            int y_coordinate = (int) (vSpacing + nodeIndex*vSpacing);

            nodesForLayer[nodeIndex] = new Node(Node_ID, x_coordinate, y_coordinate);
        }

        return nodesForLayer;
    }
}
