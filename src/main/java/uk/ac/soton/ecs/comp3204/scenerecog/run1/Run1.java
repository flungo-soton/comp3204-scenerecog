package uk.ac.soton.ecs.comp3204.scenerecog.run1;

import java.util.logging.Logger;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;

/**
 * Scene recognition classifier using KNN on tiny images.
 */
public class Run1 extends Classification<KNNAnnotator<FImage, String, FloatFV>> {

    private static final Logger LOGGER = Logger.getLogger(Run1.class.getName());

    private static final String OUTPUT = "run1.txt";

    private int k;

    public Run1(int k, DatasetUtil dsUtil, String output) {
        super(dsUtil, output);
        this.k = k;
    }

    @Override
    public KNNAnnotator<FImage, String, FloatFV> getAnnotator() {
        return new KNNAnnotator(new TinyImageExtractor(), FloatFVComparison.EUCLIDEAN, k);
    }

}
