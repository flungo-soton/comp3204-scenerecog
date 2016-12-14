package uk.ac.soton.ecs.comp3204.scenerecog;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;

/**
 * Wraps the annotator to provide a generic interface for training.
 *
 * @param <A> the type of the annotator that is wrapped.
 */
public interface AnnotatorWrapper<A extends Annotator> {

    A getAnnotator();

    void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training);
}
