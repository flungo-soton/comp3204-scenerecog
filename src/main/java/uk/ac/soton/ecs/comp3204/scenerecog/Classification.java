package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.ReadableListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.IncrementalAnnotator;

/**
 *
 * @param <A> The type of annotator that is created.
 */
public abstract class Classification<A extends IncrementalAnnotator<FImage, String>> implements Runnable {

    private static final Logger LOGGER = Logger.getLogger(Classification.class.getName());

    private final DatasetUtil dsUtil;
    private final String output;

    public Classification(DatasetUtil dsUtil, String output) {
        this.dsUtil = dsUtil;
        this.output = output;
    }

    @Override
    public void run() {
        LOGGER.log(Level.INFO, "Running classifier.");

        GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training;
        ReadableListDataset<FImage, ?> testing;
        try {
            LOGGER.log(Level.FINE, "Loading datasets");
            training = dsUtil.getTraining();
            testing = dsUtil.getTesting();
        } catch (FileSystemException ex) {
            LOGGER.log(Level.SEVERE, "Error loading datasets", ex);
            return;
        }

        GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications = trainAndClassify(training, testing);

        try {
            LOGGER.log(Level.FINE, "Writing classifications to {0}", output);
            GroupWriter<String, IdentifiableObject<FImage>> writer = new IdentifyiableGroupWriter(output);
            writer.write(classifications);
        } catch (IOException ex) {
            LOGGER.log(Level.SEVERE, "IOException writing classifications to file", ex);
        }
    }

    public abstract A getAnnotator();

    public GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>>
            trainAndClassify(
                    GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training,
                    ReadableListDataset<FImage, ?> testing
            ) {
        A annotator = getAnnotator();
        LOGGER.log(Level.FINE, "Training annotator");
        annotator.trainMultiClass(training);

        LOGGER.log(Level.FINE, "Classifying the test data");
        DatasetClassifier<String, FImage> classifier = new DatasetClassifier(annotator);
        return classifier.classifyIdentifiable(testing.toIdentifiable());
    }
}
