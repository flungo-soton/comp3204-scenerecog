package uk.ac.soton.ecs.comp3204.scenerecog.run3;


import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.BatchAnnotatorWrapper;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;
import uk.ac.soton.ecs.comp3204.scenerecog.DatasetUtil;

import java.util.logging.Logger;

public class Run3 extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    DatasetUtil datasets;

    public Run3(DatasetUtil dsUtil) {
        datasets = dsUtil;
    }

    @Override
    public BatchAnnotatorWrapper<LiblinearAnnotator<FImage, String>> getAnnotatorWrapper() {
        try {
            DenseSIFT dsift = new DenseSIFT(5, 7);
            PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 5, 7, 9, 11);

            HardAssigner<byte[], float[], IntFloatPair> assigner = PHoWExtractor.trainQuantiser(datasets.getTraining(), pdsift);

//            FeatureExtractor<DoubleFV, FImage> extractor = new PHoWExtractor(pdsift, assigner);

            // Homogeneous
            // This should give high accuracy
            // Yes, we are winning
            // Hahahahah
            // Magic, DO NOT TOUCH
            HomogeneousKernelMap homogeneousKM = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
            FeatureExtractor<DoubleFV, FImage> extractor = homogeneousKM.createWrappedExtractor(new PHoWExtractor(pdsift, assigner));

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
}
