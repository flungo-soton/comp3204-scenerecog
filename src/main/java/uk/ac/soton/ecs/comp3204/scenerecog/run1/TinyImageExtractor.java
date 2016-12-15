package uk.ac.soton.ecs.comp3204.scenerecog.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import uk.ac.soton.ecs.comp3204.scenerecog.VectorUtil;

/**
 * Extracts a tiny image from the provided image. The tiny image is produced by
 * taking the centre square of the image which is resized to a fixed size using
 * the resizer provided. The feature vector is composed of the float pixel
 * values of the reduced image.
 */
public class TinyImageExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private final ResizeProcessor resizer;

    public TinyImageExtractor(ResizeProcessor resizer) {
        this.resizer = resizer;
    }

    public TinyImageExtractor() {
        this(new ResizeProcessor(16, 16, false));
    }

    @Override
    public DoubleFV extractFeature(FImage image) {
        int minDim = Math.min(image.width, image.height);
        image = image.extractCenter(minDim, minDim);
        image = image.process(resizer);

        // Mean centering
        double[] vector = image.getDoublePixelVector();
        VectorUtil.meanCentring(vector);

        // Create feature vector then normalise
        return new DoubleFV(vector).normaliseFV(2);
    }

}
