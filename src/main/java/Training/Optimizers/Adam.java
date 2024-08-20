package Training.Optimizers;

import Structures.*;

public class Adam extends Optimizer {
    private final float beta1;
    private final float beta2;
    private final float epsilon;

    // DEFAULTS - from paper: https://arxiv.org/pdf/1412.6980
    public static final float default_beta1 = 0.9f;
    public static final float default_beta2 = 0.999f;
    public static final float default_epsilon = 1e-8f;

    public Adam(float beta1, float beta2, float epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }
    public Adam() {
        this(default_beta1, default_beta2, default_epsilon);
    }

    @Override
    public void optimize(Layer layer) {
        layer.t++;
        if (layer instanceof MLPLayer) {
            optimizeMLP((MLPLayer)layer);
        } else if (layer instanceof ConvLayer) {
            optimizeConv((ConvLayer)layer);
        } else if (layer instanceof BatchNormLayer) {
            optimizeBatchNorm((BatchNormLayer)layer);
        } else if (layer instanceof FlattenLayer) {
            // do nothing
        } else {
            throw new IllegalArgumentException("Unsupported layer type. : " + layer.getClass().getSimpleName());
        }
    }

    private void optimizeMLP(MLPLayer layer) {
        layer.t++;
        float alpha = layer.getAlpha();

        // update biased first moment estimate
        layer.m = Matrix.add(Matrix.multiply(layer.m, beta1), Matrix.multiply(layer.getGradientWeights(), 1 - beta1));
        layer.mBias = Matrix.add(Matrix.multiply(layer.mBias, beta1), Matrix.multiply(layer.getGradientBiases(), 1 - beta1));

        // update biased second raw moment estimate
        layer.v = Matrix.add(Matrix.multiply(layer.v, beta2), Matrix.multiply(Matrix.elementwiseSquare(layer.getGradientWeights()), 1 - beta2));
        layer.vBias = Matrix.add(Matrix.multiply(layer.vBias, beta2), Matrix.multiply(Matrix.elementwiseSquare(layer.getGradientBiases()), 1 - beta2));

        // compute bias corrected first moment estimate
        Matrix mHat = Matrix.divide(layer.m, (float) (1 - Math.pow(beta1, layer.t)));
        Matrix mHatBias = Matrix.divide(layer.mBias, (float) (1 - Math.pow(beta1, layer.t)));

        // compute bias corrected second raw moment estimate
        Matrix vHat = Matrix.divide(layer.v, (float) (1 - Math.pow(beta2, layer.t)));
        Matrix vHatBias = Matrix.divide(layer.vBias, (float) (1 - Math.pow(beta2, layer.t)));

        // update parameters
        layer.setWeights(Matrix.subtract(layer.getWeights(), Matrix.elementWiseDivide(Matrix.multiply(mHat, alpha), Matrix.add(Matrix.elementwiseSquareRoot(vHat), epsilon))));
        layer.setBiases(Matrix.subtract(layer.getBiases(), Matrix.elementWiseDivide(Matrix.multiply(mHatBias, alpha), Matrix.add(Matrix.elementwiseSquareRoot(vHatBias), epsilon))));
    }

    private void optimizeConv(ConvLayer layer) {
        layer.t++;
        float alpha = layer.getAlpha();

        for (int f = 0; f < layer.getNumFilters(); f++) {
            for (int d = 0; d < layer.getInputDepth(); d++) {
                for (int i = 0; i < layer.getFilterSize(); i++) {
                    for (int j = 0; j < layer.getFilterSize(); j++) {
                        // biased first moment
                        layer.m[f][d][i][j] = beta1 * layer.m[f][d][i][j] + (1 - beta1) * layer.getGradientFilters()[f][d][i][j];

                        // biased second raw moment
                        layer.v[f][d][i][j] = beta2 * layer.v[f][d][i][j] + (1 - beta2) * layer.getGradientFilters()[f][d][i][j] * layer.getGradientFilters()[f][d][i][j];

                        // bias corrected first moment estimate
                        float mHat = (float) (layer.m[f][d][i][j] / (1 - Math.pow(beta1, layer.t)));

                        // bias corrected second raw moment estimate
                        float vHat = (float) (layer.v[f][d][i][j] / (1 - Math.pow(beta2, layer.t)));

                        // update params
                        layer.filters[f][d][i][j] -= (float) (alpha * mHat / (Math.sqrt(vHat) + epsilon));
                    }
                }
            }

            layer.mBias[f] = beta1 * layer.mBias[f] + (1 - beta1) * layer.getGradientBiases()[f];
            layer.vBias[f] = beta2 * layer.vBias[f] + (1 - beta2) * layer.getGradientBiases()[f] * layer.getGradientBiases()[f];

            float mHatBias = (float) (layer.mBias[f] / (1 - Math.pow(beta1, layer.t)));
            float vHatBias = (float) (layer.vBias[f] / (1 - Math.pow(beta2, layer.t)));

            layer.biases[f] -= (float) (alpha * mHatBias / (Math.sqrt(vHatBias) + epsilon));
        }
    }

    private void optimizeBatchNorm(BatchNormLayer layer) {
        layer.t++;
        float alpha = layer.getAlpha();

        for (int d = 0; d < layer.getDepth(); d++) {
            // update biased first moment estimate
            layer.m[0][d] = Adam.default_beta1 * layer.m[0][d] + (1 - Adam.default_beta1) * layer.getGradientGamma()[d];
            layer.m[1][d] = Adam.default_beta1 * layer.m[1][d] + (1 - Adam.default_beta1) * layer.getGradientBeta()[d];

            // update biased second raw moment estimate
            layer.v[0][d] = Adam.default_beta2 * layer.v[0][d] + (1 - Adam.default_beta2) * layer.getGradientGamma()[d] * layer.getGradientGamma()[d];
            layer.v[1][d] = Adam.default_beta2 * layer.v[1][d] + (1 - Adam.default_beta2) * layer.getGradientBeta()[d] * layer.getGradientBeta()[d];

            // compute bias corrected first moment estimate
            float mHat0 = layer.m[0][d] / (1 - (float)Math.pow(Adam.default_beta1, layer.t));
            float mHat1 = layer.m[1][d] / (1 - (float)Math.pow(Adam.default_beta1, layer.t));

            // compute bias corrected second raw moment estimate
            float vHat0 = layer.v[0][d] / (1 - (float)Math.pow(Adam.default_beta2, layer.t));
            float vHat1 = layer.v[1][d] / (1 - (float)Math.pow(Adam.default_beta2, layer.t));

            // update parameters
            layer.gamma[d] -= alpha * mHat0 / (float)Math.sqrt(vHat0 + Adam.default_epsilon);
            layer.beta[d] -= alpha * mHat1 / (float)Math.sqrt(vHat1 + Adam.default_epsilon);
        }
    }
}