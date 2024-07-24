package graph6java;

public class Threshold extends Graph {
    public Threshold(String s) {
        int[] bits = new int[s.length()];
        
        for (int i=0; i<s.length(); i++)
            if (s.charAt(i)=='0')
                bits[i]=0;
            else
                bits[i]=1;
                
        initializeFromBits(bits);
    }
    
    public Threshold(int[] bits) {
        initializeFromBits(bits);
    }
    
    // Assumes that bits[] contains only zeros and ones
    public void initializeFromBits(int[] bits) {
        int n = bits.length + 1;
        int[][] A = new int[n][n];
        
        for (int i=0; i<n; i++)
            A[i][i]=0;
        
        for (int i=1; i<n; i++)
            for (int j=0; j<i; j++) {
                A[i][j] = bits[i-1];
                A[j][i] = bits[i-1];
            }
            
        initializeGraph(A);
    }
}
