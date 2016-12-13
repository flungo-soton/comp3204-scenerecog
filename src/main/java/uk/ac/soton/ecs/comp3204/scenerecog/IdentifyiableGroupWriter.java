package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import org.openimaj.data.identity.Identifiable;

/**
 * Write annotations to a using a writer using the identifiable instance's ID.
 *
 * @param <G> the type of the groups which instances belong to.
 * @param <I> the type of the instances within the groups.
 */
public class IdentifyiableGroupWriter<G, I extends Identifiable> extends GroupWriter<G, I> {

    public IdentifyiableGroupWriter(PrintStream stream) {
        super(stream);
    }

    public IdentifyiableGroupWriter(File file) throws IOException {
        super(file);
    }

    public IdentifyiableGroupWriter(String filename) throws IOException {
        super(filename);
    }

    @Override
    public String formatLine(G group, I instance) {
        return instance.getID() + " " + group.toString();
    }

}
