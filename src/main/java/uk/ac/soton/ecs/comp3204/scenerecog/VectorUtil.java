package uk.ac.soton.ecs.comp3204.scenerecog;

import org.apache.commons.math3.stat.StatUtils;

/**
 * Utilities for working with vectors.
 */
public class VectorUtil {

    public static void meanCentring(double[] vector) {
        double mean = StatUtils.mean(vector);
        VectorUtil.subtract(vector, mean);
    }

    public static void subtract(double[] vector, double val) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] -= val;
        }
    }

    private VectorUtil() {
    }

}
