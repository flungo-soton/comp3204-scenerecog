package uk.ac.soton.ecs.comp3204.scenerecog.run2;


import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.math.util.FloatArrayStatsUtils;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;

import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
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

    /**
     * Split FImage into patches.
     * Patches are mean-centred and normalised.
     * Can be use for Grid or Block-based feature.
     * @param image FImage to be split into patches.
     * @param patchSize Size of the patch. Size of N*N.
     * @param patchStep Pixels to be move in x and y directions.
     * @return List of feature vector.
     */
    private static List<float[]> getPatches(FImage image, int patchSize, int patchStep) {
        List<float[]> patches = new ArrayList<float[]>();

        for(int row=0; row<image.getHeight()-patchSize; row+=patchSize) {
            for(int col=0; col<image.getWidth()-patchSize; col+=patchSize) {
                FImage patch = image.extractROI(col, row, patchSize, patchSize);

                // Mean-centring
                float mean = FloatArrayStatsUtils.mean(image.pixels);
                patch = patch.subtract(mean);

                // Normalise
                patch.normalise();

                patches.add(patch.getFloatPixelVector());
            }
        }
        return patches;
    }
}
