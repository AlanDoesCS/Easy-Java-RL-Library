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
        double maxGradientNorm = 1.0; // You can adjust this value as needed
        clipGradients(layer, maxGradientNorm);

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

    private void clipGradients(Layer layer, double maxGradientNorm) {
        double gradientNorm = 0.0;

        if (layer instanceof MLPLayer mlpLayer) {
            gradientNorm = Math.sqrt(
                    MatrixDouble.multiply(mlpLayer.getGradientWeights(), mlpLayer.getGradientWeights().transpose()).sumOfSquares() +
                            MatrixDouble.multiply(mlpLayer.getGradientBiases(), mlpLayer.getGradientBiases().transpose()).sumOfSquares()
            );
        } else if (layer instanceof ConvLayer convLayer) {
            for (int f = 0; f < convLayer.getNumFilters(); f++) {
                for (int d = 0; d < convLayer.getInputDepth(); d++) {
                    for (int i = 0; i < convLayer.getFilterSize(); i++) {
                        for (int j = 0; j < convLayer.getFilterSize(); j++) {
                            gradientNorm += Math.pow(convLayer.getGradientFilters()[f][d][i][j], 2);
                        }
                    }
                }
                gradientNorm += Math.pow(convLayer.getGradientBiases()[f], 2);
            }
            gradientNorm = Math.sqrt(gradientNorm);
        } else if (layer instanceof BatchNormLayer bnLayer) {
            for (int d = 0; d < bnLayer.getDepth(); d++) {
                gradientNorm += Math.pow(bnLayer.getGradientGamma()[d], 2);
                gradientNorm += Math.pow(bnLayer.getGradientBeta()[d], 2);
            }
            gradientNorm = Math.sqrt(gradientNorm);
        }

        if (gradientNorm > maxGradientNorm) {
            double scalingFactor = maxGradientNorm / gradientNorm;
            if (layer instanceof MLPLayer mlpLayer) {
                mlpLayer.getGradientWeights().multiply(scalingFactor);
                mlpLayer.getGradientBiases().multiply(scalingFactor);
            } else if (layer instanceof ConvLayer convLayer) {
                for (int f = 0; f < convLayer.getNumFilters(); f++) {
                    for (int d = 0; d < convLayer.getInputDepth(); d++) {
                        for (int i = 0; i < convLayer.getFilterSize(); i++) {
                            for (int j = 0; j < convLayer.getFilterSize(); j++) {
                                convLayer.getGradientFilters()[f][d][i][j] *= scalingFactor;
                            }
                        }
                    }
                    convLayer.getGradientBiases()[f] *= scalingFactor;
                }
            } else if (layer instanceof BatchNormLayer bnLayer) {
                for (int d = 0; d < bnLayer.getDepth(); d++) {
                    bnLayer.getGradientGamma()[d] *= scalingFactor;
                    bnLayer.getGradientBeta()[d] *= scalingFactor;
                }
            }
        }
    }

    private void optimizeMLP(MLPLayer layer) {
        double alpha = layer.getAlpha();

        // update biased first moment estimate
        layer.m = MatrixDouble.add(MatrixDouble.multiply(layer.m, beta1), MatrixDouble.multiply(layer.getGradientWeights(), 1 - beta1));
        layer.mBias = MatrixDouble.add(MatrixDouble.multiply(layer.mBias, beta1), MatrixDouble.multiply(layer.getGradientBiases(), 1 - beta1));

        // update biased second raw moment estimate
        layer.v = MatrixDouble.add(MatrixDouble.multiply(layer.v, beta2), MatrixDouble.multiply(MatrixDouble.elementwiseSquare(layer.getGradientWeights()), 1 - beta2));
        layer.vBias = MatrixDouble.add(MatrixDouble.multiply(layer.vBias, beta2), MatrixDouble.multiply(MatrixDouble.elementwiseSquare(layer.getGradientBiases()), 1 - beta2));

        // compute bias corrected first moment estimate
        MatrixDouble mHat = MatrixDouble.divide(layer.m, (float) (1 - Math.pow(beta1, t)));
        MatrixDouble mHatBias = MatrixDouble.divide(layer.mBias, (float) (1 - Math.pow(beta1, t)));

        // compute bias corrected second raw moment estimate
        MatrixDouble vHat = MatrixDouble.divide(layer.v, (float) (1 - Math.pow(beta2, t)));
        MatrixDouble vHatBias = MatrixDouble.divide(layer.vBias, (float) (1 - Math.pow(beta2, t)));

        // update parameters
        layer.setWeights(MatrixDouble.subtract(layer.getWeights(), MatrixDouble.elementWiseDivide(MatrixDouble.multiply(mHat, alpha), MatrixDouble.add(MatrixDouble.elementwiseSquareRoot(vHat), epsilon))));
        layer.setBiases(MatrixDouble.subtract(layer.getBiases(), MatrixDouble.elementWiseDivide(MatrixDouble.multiply(mHatBias, alpha), MatrixDouble.add(MatrixDouble.elementwiseSquareRoot(vHatBias), epsilon))));
    }

    private void optimizeConv(ConvLayer layer) {
        double alpha = layer.getAlpha();

        for (int f = 0; f < layer.getNumFilters(); f++) {
            for (int d = 0; d < layer.getInputDepth(); d++) {
                for (int i = 0; i < layer.getFilterSize(); i++) {
                    for (int j = 0; j < layer.getFilterSize(); j++) {
                        // biased first moment
                        layer.m[f][d][i][j] = beta1 * layer.m[f][d][i][j] + (1 - beta1) * layer.getGradientFilters()[f][d][i][j];

                        // biased second raw moment
                        layer.v[f][d][i][j] = beta2 * layer.v[f][d][i][j] + (1 - beta2) * layer.getGradientFilters()[f][d][i][j] * layer.getGradientFilters()[f][d][i][j];

                        // bias corrected first moment estimate
                        float mHat = (float) (layer.m[f][d][i][j] / (1 - Math.pow(beta1, t)));

                        // bias corrected second raw moment estimate
                        float vHat = (float) (layer.v[f][d][i][j] / (1 - Math.pow(beta2, t)));

                        // update params
                        layer.filters[f][d][i][j] -= (float) (alpha * mHat / (Math.sqrt(vHat) + epsilon));
                    }
                }
            }

            layer.mBias[f] = beta1 * layer.mBias[f] + (1 - beta1) * layer.getGradientBiases()[f];
            layer.vBias[f] = beta2 * layer.vBias[f] + (1 - beta2) * layer.getGradientBiases()[f] * layer.getGradientBiases()[f];

            float mHatBias = (float) (layer.mBias[f] / (1 - Math.pow(beta1, t)));
            float vHatBias = (float) (layer.vBias[f] / (1 - Math.pow(beta2, t)));

            layer.biases[f] -= (float) (alpha * mHatBias / (Math.sqrt(vHatBias) + epsilon));
        }
    }

    private void optimizeBatchNorm(BatchNormLayer layer) {
        double alpha = layer.getAlpha();

        for (int d = 0; d < layer.getDepth(); d++) {
            // update biased first moment estimate
            layer.m[0][d] = beta1 * layer.m[0][d] + (1 - beta1) * layer.getGradientGamma()[d];
            layer.m[1][d] = beta1 * layer.m[1][d] + (1 - beta1) * layer.getGradientBeta()[d];

            // update biased second raw moment estimate
            layer.v[0][d] = beta2 * layer.v[0][d] + (1 - beta2) * layer.getGradientGamma()[d] * layer.getGradientGamma()[d];
            layer.v[1][d] = beta2 * layer.v[1][d] + (1 - beta2) * layer.getGradientBeta()[d] * layer.getGradientBeta()[d];

            // compute bias corrected first moment estimate
            double mHat0 = layer.m[0][d] / (1 - Math.pow(beta1, t));
            double mHat1 = layer.m[1][d] / (1 - Math.pow(beta1, t));

            // compute bias corrected second raw moment estimate
            double vHat0 = layer.v[0][d] / (1 - Math.pow(beta2, t));
            double vHat1 = layer.v[1][d] / (1 - Math.pow(beta2, t));

            // update parameters
            layer.gamma[d] -= alpha * mHat0 / Math.sqrt(vHat0 + epsilon);
            layer.beta[d] -= alpha * mHat1 / Math.sqrt(vHat1 + epsilon);
        }
    }
}