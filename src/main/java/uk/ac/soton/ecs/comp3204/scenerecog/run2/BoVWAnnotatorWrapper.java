package uk.ac.soton.ecs.comp3204.scenerecog.run2;

import de.bwaldvogel.liblinear.SolverType;
import java.util.ArrayList;
import java.util.List;
import org.openimaj.data.RandomData;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.AnnotatorWrapper;

/**
 * Custom annotation wrapper for Run 2 which deals with the more complex
 * requirements for training.
 */
public class BoVWAnnotatorWrapper implements AnnotatorWrapper<LiblinearAnnotator<FImage, String>> {

    private LiblinearAnnotator<FImage, String> annotator = null;

    private final int patchSize;
    private final int patchStep;
    private final int patchPerImage;
    private final int kMeans;

    public BoVWAnnotatorWrapper(int patchSize, int patchStep, int patchPerImage, int kMeans) {
        this.patchSize = patchSize;
        this.patchStep = patchStep;
        this.patchPerImage = patchPerImage;
        this.kMeans = kMeans;
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
            HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(training);
            // Build the extractor
            FeatureExtractor<DoubleFV, FImage> extractor = new BoVWExtractor(assigner, patchSize, patchStep);
            // Create the annotator
            annotator = new LiblinearAnnotator<>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001
            );
        }
        annotator.train(training);
    }

    private HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ? extends ListDataset<FImage>, FImage> datasets
    ) {
        // Use for k means cluster
        List<float[]> allPatches = new ArrayList<>();

        // From all images, pick patchPerImage random patches.
        for (FImage image : datasets) {
            List<float[]> patches = BoVWExtractor.getPatches(image, patchSize, patchStep);

            // Get patchPerImage unique patch IDs
            int[] uniqueKeys = RandomData.getUniqueRandomInts(patchPerImage, 0, patches.size());
            for (int i = 0; i < uniqueKeys.length; i++) {
                float[] patch = patches.get(uniqueKeys[i]);
                allPatches.add(patch);
            }
        }

        // Convert into float[][] so it can used for cluster
        float[][] sample = allPatches.toArray(new float[allPatches.size()][]);

        // Create 500 clusters
        FloatKMeans km = FloatKMeans.createExact(kMeans);
        FloatCentroidsResult result = km.cluster(sample);

        return result.defaultHardAssigner();
    }

}
