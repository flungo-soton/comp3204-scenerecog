package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;

/**
 * Runner for the Scene Recognition coursework. This will run all 3 recognisers.
 */
public class App {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    public static final double TRAINING_PCT = 0.85;

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
    }

    public static void run(Classification classification, String runID, DatasetUtil datasets) throws IOException {
        System.out.println("############ Run " + runID + " ############");
        System.out.println();
        CMResult<String> performance = classification.run(datasets, "run" + runID + ".txt", TRAINING_PCT);
        System.out.println(performance.getDetailReport());
        System.out.println();
    }
}
