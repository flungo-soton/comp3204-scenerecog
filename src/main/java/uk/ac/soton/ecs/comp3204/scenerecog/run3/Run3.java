package uk.ac.soton.ecs.comp3204.scenerecog.run3;

import java.util.logging.Logger;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;

public class Run3 extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    @Override
    public Run3AnnotationWrapper getAnnotatorWrapper() {
        return new Run3AnnotationWrapper();
    }
}
