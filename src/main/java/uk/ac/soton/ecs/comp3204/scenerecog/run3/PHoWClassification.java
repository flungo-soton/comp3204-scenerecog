package uk.ac.soton.ecs.comp3204.scenerecog.run3;

import java.util.logging.Logger;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import uk.ac.soton.ecs.comp3204.scenerecog.App;
import uk.ac.soton.ecs.comp3204.scenerecog.Classification;

public class PHoWClassification extends Classification<LiblinearAnnotator<FImage, String>> {

    private static final Logger LOGGER = Logger.getLogger(App.class.getName());

    @Override
    public PHoWAnnotationWrapper getAnnotatorWrapper() {
        // Extract patches from image, patches are known as bin here.
        // Extract SIFT from patches.
        // Patches are size of 7 by 7 and every 3 pixels on x and y directions.
        DenseSIFT dsift = new DenseSIFT(3, 7);
        // magFactor used for smooth
        // 0 will have no smooth
        // Scale level with 4,6,8,10
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4, 6, 8, 10);

        // Homogeneous Kernel Map
        // This should give high accuracy
        HomogeneousKernelMap homogeneousKM = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);

        return new PHoWAnnotationWrapper(pdsift, homogeneousKM);
    }
}
