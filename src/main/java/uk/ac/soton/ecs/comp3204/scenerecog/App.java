package uk.ac.soton.ecs.comp3204.scenerecog;

import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;

/**
 * Runner for the Scene Recognition coursework. This will run all 3 recognisers.
 */
public class App {
    
    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            DatasetUtil datasets = DatasetUtil.getDatasetUtil();
            
            System.out.println(datasets.getTraining());
            System.out.println(datasets.getTesting());
        } catch (FileSystemException | URISyntaxException ex) {
            LOGGER.log(Level.SEVERE, "Exception loading datasets.", ex);
        }
    }
    
}
