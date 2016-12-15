package uk.ac.soton.ecs.comp3204.scenerecog;

import java.util.Map;
import java.util.Set;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;

/**
 *
 */
public class MutlithreadCMAnalyser<OBJECT, CLASS> extends CMAnalyser<OBJECT, CLASS> {

    public MutlithreadCMAnalyser(Strategy strategy) {
        super(strategy);
    }

    @Override
    public CMResult<CLASS> analyse(
            Map<OBJECT, ClassificationResult<CLASS>> predicted,
            Map<OBJECT, Set<CLASS>> actual
    ) {
        throw new UnsupportedOperationException("Not supported yet.");
//        final List<TargetEstimatePair<CLASS, CLASS>> data = new ArrayList<TargetEstimatePair<CLASS, CLASS>>();
//
//        for (final OBJECT obj : predicted.keySet()) {
//            final Set<CLASS> pclasses = predicted.get(obj).getPredictedClasses();
//            final Set<CLASS> aclasses = actual.get(obj);
//
//            synchronized (data) {
//                strategy.add(data, pclasses, aclasses);
//            }
//        }
//
//        return new CMResult<CLASS>(eval.evaluatePerformance(data));
    }
}
