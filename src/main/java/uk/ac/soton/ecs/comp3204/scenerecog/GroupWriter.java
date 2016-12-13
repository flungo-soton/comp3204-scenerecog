package uk.ac.soton.ecs.comp3204.scenerecog;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;

/**
 * Write annotations to a using a writer.
 *
 * @param <G> the type of the groups which instances belong to.
 * @param <I> the type of the instances within the groups.
 */
public class GroupWriter<G, I> {

    private final PrintStream stream;

    public GroupWriter(PrintStream stream) {
        this.stream = stream;
    }

    public GroupWriter(File file) throws IOException {
        this(new PrintStream(file));
    }

    public GroupWriter(String filename) throws IOException {
        this(new File(filename));
    }

    public final void write(GroupedDataset<G, ? extends Dataset<I>, I> dataset) {
        for (G group : dataset.getGroups()) {
            for (I instance : dataset.getInstances(group)) {
                stream.println(formatLine(group, instance));
            }
        }
    }

    public String formatLine(G group, I instance) {
        return instance.toString() + " " + group.toString();
    }
}
