package uk.ac.soton.ecs.comp3204.scenerecog;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.IncrementalAnnotator;

/**
 *
 */
public class IncrementalAnnotatorWrapper<A extends IncrementalAnnotator> extends AnnotatorWrapper<A> {

    public IncrementalAnnotatorWrapper(A annotator) {
        super(annotator);
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {
        getAnnotator().trainMultiClass(training);
    }

}
