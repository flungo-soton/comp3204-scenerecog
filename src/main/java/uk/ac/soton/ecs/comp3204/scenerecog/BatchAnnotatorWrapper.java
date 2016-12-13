package uk.ac.soton.ecs.comp3204.scenerecog;


import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.BatchAnnotator;

/**
 * Class use to contain BatchAnnotator.
 *
 * @param <A> Type of BatchAnnotator that is created.
 */
public class BatchAnnotatorWrapper<A extends BatchAnnotator> extends AnnotatorWrapper<A> {

    public BatchAnnotatorWrapper(A annotator) {
        super(annotator);
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {
        getAnnotator().train(training);
    }
}
