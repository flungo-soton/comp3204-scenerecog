package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.output.TeeOutputStream;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;

import uk.ac.soton.ecs.comp3204.scenerecog.run1.Run1;
import uk.ac.soton.ecs.comp3204.scenerecog.run2.Run2;
import uk.ac.soton.ecs.comp3204.scenerecog.run3.Run3;

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

/*
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
            run(new Run2(), "2", datasets);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "IOException in Run 2.", e);
            return;
        }
*/

        try {
            run(new Run3(datasets), "3", datasets);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "IOException in Run 3.", e);
            return;
        }
    }

    public static void run(Classification classification, String runID, DatasetUtil datasets) throws IOException {
        PrintStream out = new PrintStream(
                new TeeOutputStream(
                        System.out,
                        new FileOutputStream("run" + runID + "_report.txt")
                )
        );
        out.println("############ Run " + runID + " ############");
        out.println();
        CMResult<String> performance = classification.run(datasets, "run" + runID + ".txt", TRAINING_PCT);
        out.println(performance.getDetailReport());
        out.println();
    }
}
