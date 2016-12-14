package uk.ac.soton.ecs.comp3204.scenerecog.run2;

import de.bwaldvogel.liblinear.SolverType;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
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
 * Custom annotation wrapper for Run 2 which deals with the more complex requirements for training.
 */
public class Run2AnnotatorWrapper implements AnnotatorWrapper<LiblinearAnnotator<FImage, String>> {

    private LiblinearAnnotator<FImage, String> annotator = null;

    private final int patchSize;
    private final int patchStep;

    public Run2AnnotatorWrapper(int patchSize, int patchStep) {
        this.patchSize = patchSize;
        this.patchStep = patchStep;
    }

    @Override
    public LiblinearAnnotator<FImage, String> getAnnotator() {
        if (annotator == null) {
            throw new IllegalStateException("Annotator must be trained first.");
        }
        return annotator;
    }

    @Override
    public void train(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training) {
        if (annotator == null) {
            // If there is no annotator, do patch extraction and initialise annotator
            HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(training, patchSize, patchStep);
            FeatureExtractor<DoubleFV, FImage> extractor = new BoVWExtractor(assigner, patchSize, patchStep);
            annotator = new LiblinearAnnotator<>(
                    extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001
            );
        }
        annotator.train(training);
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ? extends ListDataset<FImage>, FImage> datasets, int patchSize, int patchStep
    ) {
        // Use for k means cluster
        List<float[]> allPatches = new ArrayList<float[]>();

        // From all images, pick 10 random patches.
        Iterator itr = datasets.iterator();
        Random random = new Random();
        while (itr.hasNext()) {
            FImage image = (FImage) itr.next();
            List<float[]> patches = BoVWExtractor.getPatches(image, patchSize, patchStep);

            for (int i = 0; i < 10; i++) {
                float[] patch = patches.get(random.nextInt(patches.size()));
                allPatches.add(patch);
            }
        }

        // Convert into float[][] so it can used for cluster
        float[][] sample = allPatches.toArray(new float[allPatches.size()][]);

        // Create 500 clusters
        FloatKMeans km = FloatKMeans.createExact(500);
        FloatCentroidsResult result = km.cluster(sample);

        return result.defaultHardAssigner();
    }

}
