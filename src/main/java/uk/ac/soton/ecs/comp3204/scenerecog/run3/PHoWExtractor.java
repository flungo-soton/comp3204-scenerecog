package uk.ac.soton.ecs.comp3204.scenerecog.run3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

public class PHoWExtractor implements FeatureExtractor<DoubleFV, FImage> {

    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<byte[], float[], IntFloatPair> assigner;

    public PHoWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    @Override
    public DoubleFV extractFeature(FImage image) {
        pdsift.analyseImage(image);

        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

        // In every level of pyramid,
        // split image into 8 blocks (2 hor and 4 ver).
        // Then extract the features.
        PyramidSpatialAggregator<byte[], SparseIntFV> spatial
                = new PyramidSpatialAggregator<>(bovw, 2, 4);

        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}
