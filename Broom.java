package graph6java;

public class Broom extends Graph {
    public Broom(int a, int b) {
        int[][] A = new int[a+b][a+b];
        
        for (int i=0; i<a-1; i++) {
            A[i][i+1]=1;
            A[i+1][i]=1;
        }
        
        for (int i=a; i<a+b; i++) {
            A[a-1][i]=1;
            A[i][a-1]=1;
        }
        
        initializeGraph(A);
    } 
}