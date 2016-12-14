package uk.ac.soton.ecs.comp3204.scenerecog.run2;

import de.bwaldvogel.liblinear.SolverType;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.BatchAnnotatorWrapper;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;

public class Run2 extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    private static final int PATCHSIZE = 8;
    private static final int PATCHSTEP = 4;

    DatasetUtil datasets;

    public Run2(DatasetUtil dsUtil) {
        datasets = dsUtil;
    }

    @Override
    public BatchAnnotatorWrapper<LiblinearAnnotator<FImage, String>> getAnnotatorWrapper() {
        try {

            HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(datasets.getTraining());
            FeatureExtractor<DoubleFV, FImage> extractor = new BoVWExtractor(assigner, PATCHSIZE, PATCHSTEP);
            return new BatchAnnotatorWrapper<LiblinearAnnotator<FImage, String>>(
                    new LiblinearAnnotator<FImage, String>(
                            extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001
                    )
            );

        } catch (FileSystemException e) {
            e.printStackTrace();
            return null;
        }
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(VFSGroupDataset datasets) {
        // Use for k means cluster
        List<float[]> allPatches = new ArrayList<float[]>();

        // From all images, pick 10 random patches.
        Iterator itr = datasets.iterator();
        Random random = new Random();
        while (itr.hasNext()) {
            FImage image = (FImage) itr.next();
            List<float[]> patches = BoVWExtractor.getPatches(image, PATCHSIZE, PATCHSTEP);

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
