package uk.ac.soton.ecs.comp3204.scenerecog.run1;

import java.util.logging.Logger;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;
import uk.ac.soton.ecs.comp3204.scenerecog.IncrementalAnnotatorWrapper;

/**
 * Scene recognition classifier using KNN on tiny images.
 */
public class Run1 extends Classification<KNNAnnotator<FImage, String, FloatFV>> {

    private static final Logger LOGGER = Logger.getLogger(Run1.class.getName());

    private static final String OUTPUT = "run1.txt";

    private int k;

    public Run1(int k) {
        this.k = k;
    }

    @Override
    public IncrementalAnnotatorWrapper<KNNAnnotator<FImage, String, FloatFV>> getAnnotatorWrapper() {
        return new IncrementalAnnotatorWrapper<KNNAnnotator<FImage, String, FloatFV>>(
                new KNNAnnotator(new TinyImageExtractor(), FloatFVComparison.EUCLIDEAN, k)
        );
    }

}
