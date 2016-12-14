package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.ReadableListDataset;
import org.openimaj.data.identity.IdentifiableObject;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.experiment.validation.StratifiedGroupedRandomisedPercentageHoldOut;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.Annotator;

/**
 * Classify the dataset given and write the result in given file.
 *
 * @param <A> The type of annotator that is created.
 */
public abstract class Classification<A extends Annotator> {

    private static final Logger LOGGER = Logger.getLogger(Classification.class.getName());

    public CMResult<String> run(DatasetUtil dsUtil, String output, double trainingPercent) throws IOException {
        LOGGER.log(Level.INFO, "Running classifier.");

        GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training;
        ReadableListDataset<FImage, ?> testing;
        try {
            LOGGER.log(Level.FINE, "Loading datasets");
            training = dsUtil.getTraining();
            testing = dsUtil.getTesting();
        } catch (FileSystemException ex) {
            throw new IOException("Error loading datasets", ex);
        }

        // Evaluate the performance
        CMResult<String> performance = evaluate(training, trainingPercent);

        // Train and classify the test data
        GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>> classifications = trainAndClassify(training, testing);

        try {
            LOGGER.log(Level.FINE, "Writing classifications to {0}", output);
            GroupWriter<String, IdentifiableObject<FImage>> writer = new IdentifyiableGroupWriter(output);
            writer.write(classifications);
        } catch (IOException ex) {
            throw new IOException("IOException writing classifications to file", ex);
        }

        return performance;
    }

    public abstract AnnotatorWrapper<A> getAnnotatorWrapper();

    public GroupedDataset<String, ? extends Dataset<IdentifiableObject<FImage>>, IdentifiableObject<FImage>>
            trainAndClassify(
                    GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training,
                    ReadableListDataset<FImage, ?> testing
            ) {
        AnnotatorWrapper<A> annotatorWrapper = getAnnotatorWrapper();
        LOGGER.log(Level.FINE, "Training annotator");
        annotatorWrapper.train(training);

        LOGGER.log(Level.FINE, "Classifying the test data");
        DatasetClassifier<String, FImage> classifier = new DatasetClassifier(annotatorWrapper.getAnnotator());
        return classifier.classifyIdentifiableParallel(testing.toIdentifiable());
    }

    public CMResult<String> evaluate(GroupedDataset<String, ? extends ListDataset<FImage>, FImage> training, double trainingPercent) {
        LOGGER.log(Level.FINE, "Evaluating classifier performance");

        // Get the annotator
        AnnotatorWrapper<A> annotator = getAnnotatorWrapper();

        // Split the training data
        StratifiedGroupedRandomisedPercentageHoldOut<String, FImage> splits = new StratifiedGroupedRandomisedPercentageHoldOut(trainingPercent, training);

        // Train with the training data
        annotator.train(splits.getTrainingDataset());

        // Build map of validation images
        // Passing splits.getValidationDataset() results in cloned images and does not work.
        // TODO: Investigate why
        GroupedDataset<String, ListDataset<FImage>, FImage> validationDataset = splits.getValidationDataset();
        Map<FImage, Set<String>> validationMap = new HashMap<>();
        for (String group : training.getGroups()) {
            for (FImage image : validationDataset.getInstances(group)) {
                Set<String> classes = new HashSet<>();
                classes.add(group);
                validationMap.put(image, classes);
            }
        }

        // Validate the data
        ClassificationEvaluator<CMResult<String>, String, FImage> eval
                = new ClassificationEvaluator<CMResult<String>, String, FImage>(
                        annotator.getAnnotator(), validationMap, new CMAnalyser<FImage, String>(CMAnalyser.Strategy.SINGLE));

        Map<FImage, ClassificationResult<String>> guesses = eval.evaluate();
        return eval.analyse(guesses);
    }
}
