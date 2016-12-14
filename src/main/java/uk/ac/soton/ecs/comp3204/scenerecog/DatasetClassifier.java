package uk.ac.soton.ecs.comp3204.scenerecog;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.Classifier;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;

/**
 * Annotate a dataset using a provided annotator.
 *
 * @param <K> The classification class.
 * @param <V> The type of objects to be classified.
 */
public class DatasetClassifier<K, V> {

    private final Classifier<K, V> classifiier;

    public DatasetClassifier(Classifier<K, V> annotator) {
        this.classifiier = annotator;
    }

    public MapBackedDataset<K, ? extends Dataset<V>, V> classify(Dataset<V> dataset) {
        MapBackedDataset<K, ListDataset<V>, V> grouped = new MapBackedDataset<>();

        for (V object : dataset) {
            K classification = getHighestConfidence(classifiier.classify(object));
            ListDataset<V> groupDataset;
            if (grouped.containsKey(classification)) {
                groupDataset = grouped.getInstances(classification);
            } else {
                groupDataset = new ListBackedDataset<>();
                grouped.add(classification, groupDataset);
            }
            groupDataset.add(object);
        }
        return grouped;
    }

    public MapBackedDataset<K, ? extends Dataset<IdentifiableObject<V>>, IdentifiableObject<V>> classifyIdentifiable(Dataset<IdentifiableObject<V>> dataset) {
        MapBackedDataset<K, ListDataset<IdentifiableObject<V>>, IdentifiableObject<V>> grouped = new MapBackedDataset<>();
        for (IdentifiableObject<V> object : dataset) {
            K classification = getHighestConfidence(classifiier.classify(object.data));
            ListDataset<IdentifiableObject<V>> groupDataset;
            if (grouped.containsKey(classification)) {
                groupDataset = grouped.getInstances(classification);
            } else {
                groupDataset = new ListBackedDataset<>();
                grouped.add(classification, groupDataset);
            }
            groupDataset.add(object);
        }
        return grouped;
    }

    public MapBackedDataset<K, ? extends Dataset<IdentifiableObject<V>>, IdentifiableObject<V>> classifyIdentifiableParallel(Dataset<IdentifiableObject<V>> dataset) {
        final MapBackedDataset<K, ListDataset<IdentifiableObject<V>>, IdentifiableObject<V>> grouped = new MapBackedDataset<>();
        Parallel.forEach(dataset, new Operation<IdentifiableObject<V>>() {
            @Override
            public void perform(IdentifiableObject<V> object) {
                K classification = getHighestConfidence(classifiier.classify(object.data));
                ListDataset<IdentifiableObject<V>> groupDataset;
                synchronized (grouped) {
                    if (grouped.containsKey(classification)) {
                        groupDataset = grouped.getInstances(classification);
                    } else {
                        groupDataset = new ListBackedDataset<>();
                        grouped.add(classification, groupDataset);
                    }
                }
                groupDataset.add(object);
            }
        });
        return grouped;
    }

    private K getHighestConfidence(ClassificationResult<K> classificationResult) {
        K result = null;
        double resultConfidence = Double.NEGATIVE_INFINITY;
        for (K classification : classificationResult.getPredictedClasses()) {
            double confidence = classificationResult.getConfidence(classification);
            if (confidence > resultConfidence) {
                result = classification;
                resultConfidence = confidence;
            }
        }
        return result;
    }
}
