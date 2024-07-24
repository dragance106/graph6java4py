package graph6java;

public class Circulant extends Graph {
    public Circulant(int n, int[] generators, int symmetric) {
        int[][] A = new int[n][n];
        
        if (symmetric==0) {
            for (int i=0; i<n; i++)
                for (int j=0; j<generators.length; j++)
                    A[i][(i+generators[j])%n]=1;
        }
        else {
            for (int i=0; i<n; i++)
                for (int j=0; j<generators.length; j++) {
                    A[i][(i+generators[j])%n]=1;
                    A[(i+generators[j])%n][i]=1;
                }
        }
        
        initializeGraph(A);
    }    
}
