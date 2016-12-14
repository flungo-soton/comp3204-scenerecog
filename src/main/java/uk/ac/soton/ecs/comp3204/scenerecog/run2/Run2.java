package uk.ac.soton.ecs.comp3204.scenerecog.run2;

import java.util.logging.Logger;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;

public class Run2 extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    private static final int PATCHSIZE = 8;
    private static final int PATCHSTEP = 4;

    @Override
    public Run2AnnotatorWrapper getAnnotatorWrapper() {
        return new Run2AnnotatorWrapper(PATCHSIZE, PATCHSTEP);
    }

}
