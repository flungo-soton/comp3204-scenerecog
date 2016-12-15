package uk.ac.soton.ecs.comp3204.scenerecog.run3;

import de.bwaldvogel.liblinear.SolverType;
import java.util.ArrayList;
import java.util.List;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.AnnotatorWrapper;

/**
 *
 */
public class Run3AnnotationWrapper implements AnnotatorWrapper<LiblinearAnnotator<FImage, String>> {

    private final PyramidDenseSIFT<FImage> pdsift;
    private final HomogeneousKernelMap homogeneousKM;

    private LiblinearAnnotator<FImage, String> annotator = null;

    public Run3AnnotationWrapper() {
        // Extract patches from image, patches are known as bin here.
        // Extract SIFT from patches.
        // Patches are size of 7 by 7 and every 3 pixels on x and y directions.
        DenseSIFT dsift = new DenseSIFT(3, 7);
        // magFactor used for smooth
        // 0 will have no smooth
        // Scale level with 4,6,8,10
        pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4, 6, 8, 10);

        // Homogeneous Kernel Map
        // This should give high accuracy
        homogeneousKM = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
    }

    @Override
    public LiblinearAnnotator<FImage, String> getAnnotator() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    /**
     * Train the annotator. The first run will also train the hard assigner
     * using the training set provided.
     *
     * @param training the data to train with.
     */
    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {
        if (annotator == null) {
            // If there is no annotator, do patch extraction and initialise annotator
            // Construct the hard assigner
            HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(training);

            // Build the extractor wrapped in the HomogeneousKernelMap
            FeatureExtractor<DoubleFV, FImage> extractor = homogeneousKM.createWrappedExtractor(new PHoWExtractor(pdsift, assigner));
            // Create the annotator
            annotator = new LiblinearAnnotator<>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001
            );
        }
        annotator.train(training);
    }

    private HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            Dataset<FImage> sample
    ) {

        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage img : sample) {
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000) {
            allkeys = allkeys.subList(0, 10000);
        }

        // Create 600 clusters
        // Eventually there will by 600 of visual words
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

}
