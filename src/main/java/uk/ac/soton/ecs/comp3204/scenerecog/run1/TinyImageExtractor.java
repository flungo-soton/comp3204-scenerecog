package uk.ac.soton.ecs.comp3204.scenerecog.run1;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * Extracts a tiny image from the provided image. The tiny image is produced by taking the centre square of the image
 * which is resized to a fixed size using the resizer provided. The feature vector is composed of the float pixel values
 * of the reduced image.
 */
public class TinyImageExtractor implements FeatureExtractor<FloatFV, FImage> {

    private final ResizeProcessor resizer;

    public TinyImageExtractor(ResizeProcessor resizer) {
        this.resizer = resizer;
    }

    public TinyImageExtractor() {
        this(new ResizeProcessor(16, 16, false));
    }

    @Override
    public FloatFV extractFeature(FImage image) {
        int minDim = Math.min(image.getWidth(), image.getHeight());
        image = image.extractCenter(minDim, minDim);
        image = image.process(resizer);
        // TODO: zero-mean and unit length
        return new FloatFV(image.getFloatPixelVector());
    }

}
