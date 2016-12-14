package uk.ac.soton.ecs.comp3204.scenerecog;

import org.openimaj.ml.annotation.Annotator;

/**
 * Abstract class to contain annotator.
 *
 * @param <A> the type of Annotator that is wrapped.
 */
public abstract class AbstractAnnotatorWrapper<A extends Annotator> implements AnnotatorWrapper<A> {

    private final A annotator;

    public AbstractAnnotatorWrapper(A annotator) {
        this.annotator = annotator;
    }

    public final A getAnnotator() {
        return annotator;
    }

}
