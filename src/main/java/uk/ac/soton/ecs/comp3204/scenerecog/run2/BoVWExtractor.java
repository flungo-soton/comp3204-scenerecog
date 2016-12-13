package uk.ac.soton.ecs.comp3204.scenerecog.run2;


import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;


/**
 * Get all the patches from one image as features
 * Merge all the features into one
 * Return this feature as the feature of the image
 */
public class BoVWExtractor implements FeatureExtractor<DoubleFV, FImage> {


    HardAssigner<float[], float[], IntFloatPair> assigner;

    int patchSize, patchStep;

    public BoVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner, int patchSize, int patchStep) {
        this.assigner = assigner;
        this.patchSize = patchSize;
        this.patchStep = patchStep;
    }


    @Override
    public DoubleFV extractFeature(FImage object) {
        List<float[]> feature = getPatches(object, patchSize, patchStep);

        BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(this.assigner);
        // Merge all the features into one
        return bovw.aggregateVectorsRaw(feature).asDoubleFV();
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
    public static List<float[]> getPatches(FImage image, int patchSize, int patchStep) {
        List<float[]> patches = new ArrayList<float[]>();

        for(int row=0; row<image.getHeight()-patchSize; row+=patchSize) {
            for(int col=0; col<image.getWidth()-patchSize; col+=patchSize) {
                FImage patch = image.extractROI(col, row, patchSize, patchSize);

/*
                // Mean-centring
                float mean = FloatArrayStatsUtils.mean(image.pixels);
                patch = patch.subtract(mean);

                // Normalise
                patch.normalise();
*/


                patches.add(patch.getFloatPixelVector());
            }
        }
        return patches;
    }


}
