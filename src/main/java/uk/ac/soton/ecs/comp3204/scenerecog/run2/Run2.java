package uk.ac.soton.ecs.comp3204.scenerecog.run2;

import java.util.logging.Logger;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;

public class Run2 extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    private final int patchSize;
    private final int patchStep;
    private final int patchPerImage;
    private final int kMeans;

    public Run2(int patchSize, int patchStep, int patchPerImage, int kMeans) {
        this.patchSize = patchSize;
        this.patchStep = patchStep;
        this.patchPerImage = patchPerImage;
        this.kMeans = kMeans;
    }

    @Override
    public Run2AnnotatorWrapper getAnnotatorWrapper() {
        return new Run2AnnotatorWrapper(patchSize, patchStep, patchPerImage, kMeans);
    }

}
