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
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
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
public class PHoWAnnotationWrapper implements AnnotatorWrapper<LiblinearAnnotator<FImage, String>> {

    private final AbstractDenseSIFT<FImage> pdsift;
    private final HomogeneousKernelMap homogeneousKM;
    private final int maxFeatures = 10000;
    private final int visualWords = 600;

    private LiblinearAnnotator<FImage, String> annotator = null;

    public PHoWAnnotationWrapper(AbstractDenseSIFT<FImage> pdsift, HomogeneousKernelMap homogeneousKM) {
        this.pdsift = pdsift;
        this.homogeneousKM = homogeneousKM;
    }

    @Override
    public LiblinearAnnotator<FImage, String> getAnnotator() {
        if (annotator == null) {
            throw new IllegalStateException("Annotator must be trained first.");
        }
        return annotator;
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

        // Extract the SIFT features
        for (FImage img : sample) {
            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        // Truncate the number of features if too many
        if (allkeys.size() > maxFeatures) {
            allkeys = allkeys.subList(0, maxFeatures);
        }

        // Create 600 clusters
        // Eventually there will by 600 of visual words
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(visualWords);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

}
