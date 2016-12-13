package uk.ac.soton.ecs.comp3204.scenerecog;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;

/**
 * Abstract class to contain annotator.
 *
 * @param <A> The type of Annotator that is created.
 */
public abstract class AnnotatorWrapper<A extends Annotator> {

    private final A annotator;

    public AnnotatorWrapper(A annotator) {
        this.annotator = annotator;
    }

    public final A getAnnotator() {
        return annotator;
    }

    public abstract void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training);

}
