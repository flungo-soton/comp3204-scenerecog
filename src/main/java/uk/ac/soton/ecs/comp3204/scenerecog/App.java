package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.output.TeeOutputStream;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.ReadableListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import uk.ac.soton.ecs.comp3204.scenerecog.run1.Run1;
import uk.ac.soton.ecs.comp3204.scenerecog.run2.Run2;
import uk.ac.soton.ecs.comp3204.scenerecog.run3.Run3;

/**
 * Runner for the Scene Recognition coursework. This will run all 3 recognisers.
 */
public class App {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    public static final double TRAINING_PCT = 0.85;

    private static final int BOVW_PATCH_SIZE = 8;
    private static final int BOVW_PATCH_STEP = 4;
    private static final int BOVW_PATCHES_PER_IMAGE = 10;
    private static final int BOVW_K_MEANS = 500;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        DatasetUtil datasets;
        try {
            datasets = DatasetUtil.getDatasetUtil();

            LOGGER.log(Level.CONFIG, "Training dataset: {0}", datasets.getTraining());
            LOGGER.log(Level.CONFIG, "Testing dataset: {0}", datasets.getTesting());

        } catch (FileSystemException | URISyntaxException ex) {
            LOGGER.log(Level.SEVERE, "Exception loading datasets.", ex);
            return;
        }

        try {
            // Run, Run1
            run(new Run1(1), "1", datasets);
            run(new Run1(3), "1-3", datasets);
            run(new Run1(5), "1-5", datasets);
            run(new Run1(11), "1-11", datasets);
        } catch (IOException ex) {
            LOGGER.log(Level.SEVERE, "IOException running classification run 1.", ex);
        }
        try {
            // Run, Run2
            run(new Run2(BOVW_PATCH_SIZE, BOVW_PATCH_STEP, BOVW_PATCHES_PER_IMAGE, BOVW_K_MEANS), "2", datasets);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "IOException running classification run 2.", e);
        }
        try {
            // Run, Run3
            run(new Run3(datasets), "3", datasets);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "IOException running classification run 3.", e);
        }
    }

    /**
     * Run a classification to create an evaluation report and the
     * classification of the test data.
     *
     * @param classification the classification to run.
     * @param runID the unique ID of the run, to identify the output files.
     * @param datasets the dataset utility which provides the training and
     * testing data.
     * @throws IOException if there is an error reading the datasets or writing
     * the reports.
     */
    public static void run(Classification classification, String runID, DatasetUtil datasets) throws IOException {
        LOGGER.log(Level.INFO, "Running classifier: {0}", runID);

        // Files
        String evaluationOutput = "run" + runID + "_report.txt";
        String classificationOutput = "run" + runID + ".txt";

        // Setup a split output to console and file.
        PrintStream out = new PrintStream(
                new TeeOutputStream(
                        System.out,
                        new FileOutputStream(evaluationOutput)
                )
        );

        // Load the training and testing data.
        GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training;
        ReadableListDataset<FImage, ?> testing;
        try {
            LOGGER.log(Level.FINE, "Loading datasets");
            training = datasets.getTraining();
            testing = datasets.getTesting();
        } catch (FileSystemException ex) {
            throw new IOException("Error loading datasets", ex);
        }

        out.println("############ Run " + runID + " ############");
        out.println();

        // Evaluate the performance
        LOGGER.log(Level.FINE, "Evaluating the performance of the classifier");
        CMResult<String> performance = classification.evaluate(training, TRAINING_PCT);
        out.println(performance.getDetailReport());
        out.println();

        // Do the classification
        // Train and classify the test data
        LOGGER.log(Level.FINE, "Training and classifying testing data");
        GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications
                = classification.trainAndClassify(training, testing);

        try {
            LOGGER.log(Level.FINE, "Writing classifications to {0}", classificationOutput);
            GroupWriter<String, IdentifiableObject<FImage>> writer = new IdentifyiableGroupWriter(classificationOutput);
            writer.write(classifications);
        } catch (IOException ex) {
            throw new IOException("IOException writing classifications to file", ex);
        }
    }
}
