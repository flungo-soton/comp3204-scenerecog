package uk.ac.soton.ecs.comp3204.scenerecog.run2;


import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;

import java.net.URISyntaxException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Run2 {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    public static void main(String[] args) {
        try {
            DatasetUtil datasets = DatasetUtil.getDatasetUtil();

            System.out.println(datasets.getTraining());
            System.out.println(datasets.getTesting());

            HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(datasets.getTraining());

            FeatureExtractor<DoubleFV, FImage> extractor = new BoVWExtractor(assigner);

            LiblinearAnnotator<FImage, String> ann = new LiblinearAnnotator<FImage, String>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        } catch (FileSystemException | URISyntaxException ex) {
            LOGGER.log(Level.SEVERE, "Exception loading datasets.", ex);
        }
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset datasets) {


        return null;
    }

    static class BoVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<float[], float[], IntFloatPair> assigner;

        public BoVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage object) {


            return null;
        }
    }


}
