package graph6java;
import java.io.*;

public class SubsetTemplate {
    // Variables needed to run the template
    private String g6code;          // g6code of a graph
    private Graph g;                // graph
       
    // Files
    private BufferedReader in;      // input file with graphs
    private PrintWriter outResults; // output file for selected graphs and other data

    // graph counter, useful for occassionally printing the number of graphs processed so far
    private static int counter;

    private int n;
    private int m;
    private int[] deg;
    private int bigDelta;
    private int smallDelta;
    private double[] eig;
    private double energy;
    
    public SubsetTemplate() {
    }

    public boolean counterexample(Graph g) {
        // Calculate necessary invariants here:
        energy = g.energy();
        deg = g.degrees();
        n = g.n();
        
        // find bigDelta and smallDelta first
        bigDelta = 0;
        smallDelta = n;
        for (int i=0; i<n; i++) {
            if (deg[i]>bigDelta)
                bigDelta = deg[i];
            if (deg[i]<smallDelta)
                smallDelta = deg[i];
        }
        
        // If energy is greater than bigDelta+smallDelta, then who cares...
        if (DoubleUtil.compareTo(energy - bigDelta - smallDelta, 0)>0)
            return false;
            
        // Ok, bigDelta+smallDelta is smaller than energy now. 
        // Is the graph non-singular?
        eig = g.Aspectrum();
        for (int i=0; i<n; i++)
            if (DoubleUtil.equals(eig[i], 0))
                return false;       // singular graphs do not count!
                 
        return true;    // yes, this is what we've been looking for!
    }
    
    /** 
     * The main method whose argument inputFileName
     * points to a file containing graphs in g6 format,
     * while createDotFiles instructs whether to write Graphviz .dot files for g6codes
     */
    public void run(String inputFileName, int createDotFiles) throws IOException {
        long startTime = System.currentTimeMillis();               // Take a note of starting time
        counter = 0;                                               // Initialise counter
                
        in = new BufferedReader(new FileReader(inputFileName));    // Open input and output files
        outResults = new PrintWriter(new BufferedWriter(new FileWriter(inputFileName + "-results.csv")));
        // note that we are not giving header row to this csv file!
        
        g6code = new String();        
        while ((g6code = in.readLine())!=null) {  // Loading g6 codes until an empty line is found
            g = new Graph(g6code);                // Create a graph out of its g6 code
            
            // is it a counterexample to the frustrating energy conjecture?
            if (counterexample(g)) {
                // output g6code, Delta, delta, energy, and eigenvalues (with all decimals)
                outResults.print(g6code + ", " + bigDelta + ", " + smallDelta + ", " + energy);
                for (int i=0; i<g.n(); i++)
                    outResults.printf(", " + eig[i]);
                outResults.println();
                
                // export graph in Graphviz format for later visualisation
                if (createDotFiles!=0)
                    g.saveDotFormat("frust-n-" + g.n() + "-g6code-" + g6code + ".dot");
            }
            
            counter++;                            // Update counter and report progress
            if (counter % 1000000 == 0)
                System.err.println(inputFileName + ": " + counter + " graphs  processed so far...");
        }
        
        in.close();                              // Testing done, close the files
        outResults.close();
        
        long totalTime = System.currentTimeMillis() - startTime;    // Report elapsed time
        System.out.println("Time elapsed: " + 
            (totalTime / 60000) + " min, " + ((double) (totalTime % 60000) / 1000) + " sec");
    }
    
    // This function may be used to run the template from out of BlueJ
    public static void main(String[] args) throws IOException, NumberFormatException {
        new SubsetTemplate().run(args[0], Integer.decode(args[1]));
    }
}