package Training.Optimizers;

import Structures.*;

public class Adam extends Optimizer {
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    private final double learningRateDecay;
    private final double learningRateMin;
    private final double lambda; // Regularization strength

    // DEFAULTS - from paper: https://arxiv.org/pdf/1412.6980
    public static final double default_beta1 = 0.9;
    public static final double default_beta2 = 0.999;
    public static final double default_epsilon = 1e-8;
    public static final double default_lambda = 0.0; // Set to 0 if not using regularization

    // Timestep for bias correction
    private int timestep;

    // For tracking beta powers to avoid large exponentiation
    private double beta1Power;
    private double beta2Power;

    public Adam(double beta1, double beta2, double epsilon, double learningRateDecay, double learningRateMin, double lambda) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.learningRateDecay = learningRateDecay;
        this.learningRateMin = learningRateMin;
        this.lambda = lambda;
        this.timestep = 0;
        this.beta1Power = 1.0;
        this.beta2Power = 1.0;
    }

    public Adam() {
        this(default_beta1, default_beta2, default_epsilon, 1.0, 0.0, default_lambda);
    }

    @Override
    public void optimize(Layer layer) {
        timestep++;
        beta1Power *= beta1;
        beta2Power *= beta2;

        double maxGradientNorm = 1.0;
        clipGradients(layer, maxGradientNorm);

        if (layer instanceof MLPLayer) {
            optimizeMLP((MLPLayer) layer);
        } else if (layer instanceof ConvLayer) {
            optimizeConv((ConvLayer) layer);
        } else if (layer instanceof BatchNormLayer) {
            optimizeBatchNorm((BatchNormLayer) layer);
        } else if (layer instanceof FlattenLayer) {
            // FlattenLayer doesn't have weights to optimize, so skip it
        } else {
            throw new IllegalArgumentException("Unsupported layer type: " + layer.getClass().getSimpleName());
        }
    }

    private void clipGradients(Layer layer, double maxGradientNorm) {
        double gradientNorm = 0.0;

        if (layer instanceof MLPLayer mlpLayer) {
            gradientNorm = Math.sqrt(
                    mlpLayer.getGradientWeights().sumOfSquares() +
                            mlpLayer.getGradientBiases().sumOfSquares()
            );
        } else if (layer instanceof ConvLayer convLayer) {
            double sumSquares = 0.0;
            for (int f = 0; f < convLayer.getNumFilters(); f++) {
                for (int d = 0; d < convLayer.getInputDepth(); d++) {
                    for (int i = 0; i < convLayer.getFilterSize(); i++) {
                        for (int j = 0; j < convLayer.getFilterSize(); j++) {
                            double grad = convLayer.getGradientFilters()[f][d][i][j];
                            sumSquares += grad * grad;
                        }
                    }
                }
                double gradBias = convLayer.getGradientBiases()[f];
                sumSquares += gradBias * gradBias;
            }
            gradientNorm = Math.sqrt(sumSquares);
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
            }
        }
    }

    private void optimizeMLP(MLPLayer layer) {
        // Adjust alpha
        double alpha = layer.getAlpha();
        alpha = Math.max(alpha * learningRateDecay, learningRateMin);
        layer.setAlpha(alpha);

        // L2 regularization
        if (lambda > 0.0) {
            MatrixDouble regularizationTerm = MatrixDouble.multiply(layer.getWeights(), lambda);
            layer.getGradientWeights().add(regularizationTerm);
        }

        // Update biased first moment estimate
        layer.mW = MatrixDouble.add(
                MatrixDouble.multiply(layer.mW, beta1),
                MatrixDouble.multiply(layer.getGradientWeights(), 1 - beta1)
        );
        layer.mB = MatrixDouble.add(
                MatrixDouble.multiply(layer.mB, beta1),
                MatrixDouble.multiply(layer.getGradientBiases(), 1 - beta1)
        );

        // Update biased second raw moment estimate
        layer.vW = MatrixDouble.add(
                MatrixDouble.multiply(layer.vW, beta2),
                MatrixDouble.multiply(MatrixDouble.elementwiseSquare(layer.getGradientWeights()), 1 - beta2)
        );
        layer.vB = MatrixDouble.add(
                MatrixDouble.multiply(layer.vB, beta2),
                MatrixDouble.multiply(MatrixDouble.elementwiseSquare(layer.getGradientBiases()), 1 - beta2)
        );

        // Compute bias-corrected first moment estimate
        MatrixDouble mHatW = MatrixDouble.divide(layer.mW, 1 - beta1Power);
        MatrixDouble mHatB = MatrixDouble.divide(layer.mB, 1 - beta1Power);

        // Compute bias-corrected second raw moment estimate
        MatrixDouble vHatW = MatrixDouble.divide(layer.vW, 1 - beta2Power);
        MatrixDouble vHatB = MatrixDouble.divide(layer.vB, 1 - beta2Power);

        // Update weights and biases with Adam optimization
        MatrixDouble weightUpdates = MatrixDouble.elementWiseDivide(
                mHatW,
                MatrixDouble.add(MatrixDouble.elementwiseSquareRoot(vHatW), epsilon)
        );
        MatrixDouble biasUpdates = MatrixDouble.elementWiseDivide(
                mHatB,
                MatrixDouble.add(MatrixDouble.elementwiseSquareRoot(vHatB), epsilon)
        );

        layer.setWeights(
                MatrixDouble.subtract(layer.getWeights(), MatrixDouble.multiply(weightUpdates, alpha))
        );
        layer.setBiases(
                MatrixDouble.subtract(layer.getBiases(), MatrixDouble.multiply(biasUpdates, alpha))
        );
    }

    private void optimizeConv(ConvLayer layer) {
        double alpha = layer.getAlpha();
        alpha = Math.max(alpha * learningRateDecay, learningRateMin);
        layer.setAlpha(alpha);

        // L2 regularization
        if (lambda > 0.0) {
            for (int f = 0; f < layer.getNumFilters(); f++) {
                for (int d = 0; d < layer.getInputDepth(); d++) {
                    for (int i = 0; i < layer.getFilterSize(); i++) {
                        for (int j = 0; j < layer.getFilterSize(); j++) {
                            layer.getGradientFilters()[f][d][i][j] += lambda * layer.filters[f][d][i][j];
                        }
                    }
                }
            }
        }

        for (int f = 0; f < layer.getNumFilters(); f++) {
            for (int d = 0; d < layer.getInputDepth(); d++) {
                for (int i = 0; i < layer.getFilterSize(); i++) {
                    for (int j = 0; j < layer.getFilterSize(); j++) {
                        double grad = layer.getGradientFilters()[f][d][i][j];

                        layer.mFilters[f][d][i][j] = beta1 * layer.mFilters[f][d][i][j] + (1 - beta1) * grad;
                        layer.vFilters[f][d][i][j] = beta2 * layer.vFilters[f][d][i][j] + (1 - beta2) * grad * grad;

                        // Bias-corrected first & second moment estimate
                        double mHat = layer.mFilters[f][d][i][j] / (1 - beta1Power);
                        double vHat = layer.vFilters[f][d][i][j] / (1 - beta2Power);

                        // Update filter weights
                        layer.filters[f][d][i][j] -= alpha * mHat / (Math.sqrt(vHat) + epsilon);
                    }
                }
            }

            // Update biases
            double gradBias = layer.getGradientBiases()[f];
            layer.mBiases[f] = beta1 * layer.mBiases[f] + (1 - beta1) * gradBias;
            layer.vBiases[f] = beta2 * layer.vBiases[f] + (1 - beta2) * gradBias * gradBias;

            double mHatBias = layer.mBiases[f] / (1 - beta1Power);
            double vHatBias = layer.vBiases[f] / (1 - beta2Power);

            layer.biases[f] -= alpha * mHatBias / (Math.sqrt(vHatBias) + epsilon);
        }
    }

    private void optimizeBatchNorm(BatchNormLayer layer) {
        double alpha = layer.getAlpha();
        alpha = Math.max(alpha * learningRateDecay, learningRateMin);
        layer.setAlpha(alpha);

        for (int d = 0; d < layer.getDepth(); d++) {
            // Gamma
            double gradGamma = layer.getGradientGamma()[d];
            layer.m[0][d] = beta1 * layer.m[0][d] + (1 - beta1) * gradGamma;
            layer.v[0][d] = beta2 * layer.v[0][d] + (1 - beta2) * gradGamma * gradGamma;

            double mHatGamma = layer.m[0][d] / (1 - beta1Power);
            double vHatGamma = layer.v[0][d] / (1 - beta2Power);

            layer.gamma[d] -= alpha * mHatGamma / (Math.sqrt(vHatGamma) + epsilon);

            // Beta
            double gradBeta = layer.getGradientBeta()[d];
            layer.m[1][d] = beta1 * layer.m[1][d] + (1 - beta1) * gradBeta;
            layer.v[1][d] = beta2 * layer.v[1][d] + (1 - beta2) * gradBeta * gradBeta;

            double mHatBeta = layer.m[1][d] / (1 - beta1Power);
            double vHatBeta = layer.v[1][d] / (1 - beta2Power);

            layer.beta[d] -= alpha * mHatBeta / (Math.sqrt(vHatBeta) + epsilon);
        }
    }
}