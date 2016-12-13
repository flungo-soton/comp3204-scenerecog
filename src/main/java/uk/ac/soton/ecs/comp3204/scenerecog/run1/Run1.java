package uk.ac.soton.ecs.comp3204.scenerecog.run1;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ReadableListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.FloatFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetClassifier;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;
import uk.ac.soton.ecs.comp3204.scenerecog.GroupWriter;
import uk.ac.soton.ecs.comp3204.scenerecog.IdentifyiableGroupWriter;

/**
 * Scene recognition classifier using KNN on tiny images.
 */
public class Run1 implements Runnable {

    private static final Logger LOGGER = Logger.getLogger(Run1.class.getName());

    private static final String OUTPUT = "run1.txt";

    @Override
    public void run() {
        LOGGER.log(Level.INFO, "Running Run1 classifier.");
        DatasetUtil dsUtil;
        GroupedDataset training;
        ReadableListDataset testing;
        try {
            LOGGER.log(Level.FINE, "Loading datasets");
            dsUtil = DatasetUtil.getDatasetUtil();
            training = dsUtil.getTraining();
            testing = dsUtil.getTesting();
        } catch (URISyntaxException | FileSystemException ex) {
            LOGGER.log(Level.SEVERE, "Error loading datasets", ex);
            return;
        }

        LOGGER.log(Level.FINER, "Initialising annotator");
        KNNAnnotator<FImage, String, FloatFV> annotator = new KNNAnnotator(new TinyImageExtractor(), FloatFVComparison.EUCLIDEAN, training.size());

        LOGGER.log(Level.FINE, "Training annotator");
        annotator.trainMultiClass(training);

        LOGGER.log(Level.FINE, "Classifying the test data");
        DatasetClassifier<String, FImage> classifier = new DatasetClassifier(annotator);
        GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications = classifier.classifyIdentifiable(testing.toIdentifiable());

        try {
            LOGGER.log(Level.FINE, "Writing classifications to {0}", OUTPUT);
            GroupWriter<String, IdentifiableObject<FImage>> writer = new IdentifyiableGroupWriter(OUTPUT);
            writer.write(classifications);
        } catch (IOException ex) {
            LOGGER.log(Level.SEVERE, "IOException writing classifications to file", ex);
        }
    }

}
