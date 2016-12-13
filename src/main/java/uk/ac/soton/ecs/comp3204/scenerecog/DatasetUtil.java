package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.ImageUtilities;

/**
 * Utility for loading the testing and training datasets (as zips) from a given directory. The zips should be named
 * {@code training.zip} and {@code testing.zip}.
 */
public final class DatasetUtil {

    private static final Logger LOGGER = Logger.getLogger(DatasetUtil.class.getName());

    public static final String REMOTE_DATASET_DIRECTORY = "http://comp3204.ecs.soton.ac.uk/cw/";
    public static final String TRAINING_DATASET = "training.zip!/training";
    public static final String TESTING_DATASET = "testing.zip!/testing";

    private URI directory;

    public DatasetUtil(URI directory) {
        this.directory = directory;
    }

    public VFSGroupDataset getTraining() throws FileSystemException {
        return new VFSGroupDataset("zip:" + directory.resolve(TRAINING_DATASET).toString(), ImageUtilities.FIMAGE_READER);
    }

    public VFSListDataset getTesting() throws FileSystemException {
        return new VFSListDataset("zip:" + directory.resolve(TESTING_DATASET).toString(), ImageUtilities.FIMAGE_READER);
    }

    public static DatasetUtil getDatasetUtil() throws URISyntaxException {
        File localFile = new File("data").getAbsoluteFile();
        if (localFile.isDirectory()) {
            LOGGER.log(Level.CONFIG, "'data' directory exists, datasets will be loaded from that directory.");
            return new DatasetUtil(localFile.toURI());
        } else {
            LOGGER.log(Level.CONFIG, "'data' directory does not exists, datasets will be loaded from {0}", REMOTE_DATASET_DIRECTORY);
            return new DatasetUtil(new URI(REMOTE_DATASET_DIRECTORY));
        }
    }

}
