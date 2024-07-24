/**
 * Graph.java
 * 
 * This class creates adjacency matrix of a graph from its g6 code.
 * Values of invariants values are obtained by calling appropriate functions.
 * 
 * @author Dragan Stevanovic, Mohammad Ghebleh, Ali Kanso
 * @version April 2018
 */
package graph6java;  

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.EigenOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.interfaces.decomposition.EigenDecomposition_F64;

import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.alg.isomorphism.VF2GraphIsomorphismInspector;
import org.jgrapht.alg.matching.DenseEdmondsMaximumCardinalityMatching;
import org.jgrapht.alg.interfaces.MatchingAlgorithm;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import java.util.Set;
import java.util.Arrays;
import java.util.Comparator;

public class Graph
{
    private int[] bits;         // Sequence of bits reconstructed from g6 code
    private int n;              // number of vertices (order)
    private int m;              // number of edges (size)
    private int[] degree;       // degree sequence
    private int[][] A;          // adjacency matrix
    
    ////////////////////////////////////
    // CONSTRUCTORS AND BASIC METHODS //
    ////////////////////////////////////
    
    /**
     * Empty constructor, needed for subclassing
     */
    public Graph() {
        n=0;
    }
    
    /**
     * Constructor of a graph from g6 code
     */
    public Graph(String s) {
        n=s.charAt(0)-63;       // number of vertices is obtained from the first character of g6 code
        int firsti=1;        
        if (s.charAt(0)>=126) {
            n=(s.charAt(1)-63)*4096 + (s.charAt(2)-63)*64 + (s.charAt(3)-63);
            firsti=4;
        }
        
        int bindex = 0;         // transform g6 code characters into bit sequence
        bits = new int[6*s.length()];
        for (int i=firsti; i<s.length(); i++) {
            int k = s.charAt(i)-63;
            for (int j=0; j<=5; j++) {
                bits[bindex+5-j] = k%2;
                k = k/2;
            }
            bindex += 6;
        }
        
        A = new int[n][n];        // initialize adjacency matrix, degree sequence and number of edges
        degree = new int[n];      // indexing always starts at 0  
        for (int i=0; i<n; i++)
            degree[i] = 0;
        m = 0;

        bindex = 0;               // processes bit sequence to fill up adjacency matrix, degree sequence and number of edges
        for (int j=1; j<n; j++)
            for (int i=0; i<j; i++) {
                A[i][j] = bits[bindex];
                A[j][i] = bits[bindex];

                degree[i] += bits[bindex];
                degree[j] += bits[bindex];
                
                m += bits[bindex];
                
                bindex++;
            }                    
    }

    /**
     * Returns g6 representation of a graph constructed from its adjacency matrix.
     */
    public String toGraph6() {
        StringBuffer s = new StringBuffer();
        
        // encode the number of vertices
        if (n<63) {
            char a = (char) (n+63);
            s.append(a);
        }
        else {
            char a = 126;
            char b = (char) (n/4096);
            char c = (char) ((n%4096)/64);
            char d = (char) (n%64);
            s.append(new char[]{a,b,c,d});
        }
                
        // adjacency matrix entries go into bits sequence
        char[] newbits = new char[n*(n-1)/2+6];
        for (int i=0; i<newbits.length; i++)
            newbits[i]=0;
            
        int blength = 0;
        for (int j=1; j<n; j++)
            for (int i=0; i<j; i++) {
                newbits[blength] = (char) (A[i][j]);
                blength++;
            }               

        // bits sequence entries go into characters
        int bindex = 0;
        while (bindex<blength) {
            char k = (char) (63 + 32*newbits[bindex] 
                                + 16*newbits[bindex+1] 
                                +  8*newbits[bindex+2]
                                +  4*newbits[bindex+3] 
                                +  2*newbits[bindex+4] 
                                +    newbits[bindex+5]);
            s.append(k);
            bindex += 6;
        }
            
        return s.toString();
    }

    /**
     *  Constructor of a graph from provided adjacency matrix
     *  May be used to create graph complement or results of other graph operations
     *  Assumption: A is a symmetric, (0,1)-matrix
     */
    public Graph(int A[][]) {
        initializeGraph(A);
    }
    
    public void initializeGraph(int A[][]) {
        this.A = A;                 // adjacency matrix entries do not get copied, 
                                    // only pointer to the matrix gets copied
        n = A.length;               // number of vertices
        
        degree = new int[n];        // initializes degrees and the number of edges
        for (int i=0; i<n; i++)
            degree[i] = 0;
        m = 0;

        for (int i=0; i<n; i++)     // processes degrees and the number of edges
            for (int j=0; j<i; j++) 
                if (A[i][j]==1) {
                    degree[i]++;
                    degree[j]++;
                    m++;
                }
    }

    /** 
     * Methods returning values of numbers of vertices, edges, degrees and adjacency matrix
     */
    public int n() {
        return n;
    }

    public int m() {
        return m;
    }
    
    public int[] degrees() {
        return degree;
    }

    public double[] averageDegrees() {
        double[] ad = new double[n];
        for (int i=0; i<n; i++) {
            ad[i] = 0.0;
            for (int j=0; j<n; j++)
                if (A[i][j]==1)
                    ad[i] += degree[j];
            ad[i] /= degree[i];
        }
        
        return ad;
    }
    
    public int[][] Amatrix() {
        return A;
    }

    public Graph complement() {
        int[][] B = new int[n][n];
        
        for (int u=0; u<n; u++)
            for (int v=u+1; v<n; v++) {
                B[u][v] = 1-A[u][v];
                B[v][u] = 1-A[v][u];
            }
            
        return new Graph(B);
    }
    
    ///////////////////
    // Connectedness //
    ///////////////////
    
    public boolean isConnected() {
        return numberComponents()==1;
    }
    
    public int numberComponents() {
        int[] component = new int[n];
        int[] stack = new int[n*n];
        
        int current = 0;
        for (int v=0; v<n; v++)
            if (component[v]==0) {
                current++;
                
                // DFS_iterative:
                int stackPointer = -1;            // let S be a stack
                stack[++stackPointer]=v;          // S.push(v)
                while (stackPointer>=0) {         // while S is not empty do
                    int u = stack[stackPointer--];    // v = S.pop()
                    if (component[u]==0) {            // if v is not labeled as discovered then
                        component[u]=current;             // label v as discovered
                        for (int w=0; w<n; w++)           // for all edges from v to w in G.adjacentEdges(v) do 
                            if ((A[u][w]==1)&&(component[w]==0))
                                stack[++stackPointer]=w;  // S.push(w)
                    }
                }
            }
            
        return current;
    }
    
    //////////////////////
    // SPECTRAL METHODS //
    //////////////////////
    
    /**
     * Laplacian matrix
     */
    private int[][] L;
    private boolean LExists = false;
    
    public int[][] Lmatrix() {
        if (LExists) 
            return L;
        
        LExists = true;        
        L = new int[n][n];

        for (int i=0; i<n; i++)     // off-diagonal entries are opposite of adjacency matrix entries
            for (int j=0; j<i; j++) {
                L[i][j] = -A[i][j];
                L[j][i] = -A[j][i];
            }
            
        for (int i=0; i<n; i++)     // diagonal entries are equal to corresponding degrees
            L[i][i] = degree[i];
        
        return L;
    }
    
    /** 
     * Signless Laplacian matrix
     */
    private int[][] Q;
    private boolean QExists = false;
    
    public int[][] Qmatrix() {
        if (QExists)
            return Q;
            
        QExists = true;
        Q = new int[n][n];
        
        for (int i=0; i<n; i++)     // off-diagonal entries are equal to adjacency matrix entries
            for (int j=0; j<i; j++) {
                Q[i][j] = A[i][j];
                Q[j][i] = A[j][i];
            }
            
        for (int i=0; i<n; i++)     // diagonal entries are equal to corresponding degrees
            Q[i][i] = degree[i];
            
        return Q;    
    }
    
    /** 
     * Distance matrix by Floyd-Warshall algorithm
     */
    private int[][] D;
    private boolean DExists = false;
    
    public int[][] Dmatrix() {
        if (DExists)
            return D;
            
        DExists = true;
        D = new int[n][n];
        
        for (int i=0; i<n; i++)          // initializes distance matrix
            for (int j=0; j<n; j++)
                if (i==j) D[i][j]=0;
                else if (A[i][j]==1) 
                          D[i][j]=1;
                     else D[i][j]=n;
                     
        for (int k=0; k<n; k++)          // the main loop of the Floyd-Warshall algorithm
            for (int i=0; i<n; i++)
                for (int j=0; j<n; j++)
                    if (D[i][j] > D[i][k] + D[k][j])
                        D[i][j] = D[i][k] + D[k][j];
                        
        return D;
    }
    
    /**
     * Distance Laplacian matrix
     */
    private int[][] DL;
    private boolean DLExists = false;
    
    public int[][] DLmatrix() {
        if (DLExists)
            return DL;
            
        DLExists = true;
        DL = new int[n][n];
        
        int[] trans = transmissions();
        Dmatrix();
        
        for (int i=0; i<n; i++)
            DL[i][i] = trans[i];
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++) {
                DL[i][j] = -D[i][j];
                DL[j][i] = -D[j][i];
            }
        
        return DL;
    }
    
    /** 
     * Modularity matrix
     */
    private double[][] M;
    private boolean MExists = false;
    
    public double[][] Mmatrix() {
        if (MExists)
            return M;
            
        MExists = true;    
        M = new double[n][n];
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                M[i][j] = ((double) A[i][j]) - ((double) degree[i]*degree[j])/(2*m);

        return M;
    }

    /*
     * Adjacency characteristic polynomial: 
     * Acharpoly()[0] is the highest coefficient, Acharpoly[n] is the lowest coefficient.
     */
    public long[] Acharpoly() {
        return Graph.charpoly(A);
    }
    
    public long[] Lcharpoly() {
        Lmatrix();
        return Graph.charpoly(L);
    }
    
    public long[] Qcharpoly() {
        Qmatrix();
        return Graph.charpoly(Q);
    }
    
    public long[] Dcharpoly() {
        Dmatrix();
        return Graph.charpoly(D);
    }
    
    public long[] DLcharpoly() {
        DLmatrix();
        return Graph.charpoly(DL);
    }
    
    public double[] Mcharpoly() {
        Mmatrix();
        return Graph.charpoly(M);
    }
    
    /**
     * Iterative implementation of Berkowitz's algorithm
     * for computing coefficients of the characteristic polynomial of an integer matrix.
     */
    public static long[] charpoly(int[][] A) {
        int n = A.length;               // dimension of A
        int i;                          // shows which diagonal entry of A is currently being processed

        // was this an empty matrix?
        if (n==0) {
            long[] z = new long[1];
            z[0]=1;
            return z;
        }

        // done separately for n-1
        long[][] v = new long[2][1];
        v[0][0] = 1;
        v[1][0] = -A[n-1][n-1];
        
        // going backwards from n-2 to 0
        for (i=n-2; i>=0; i--) {
            int k = n-i;                         // dimension of currently processed submatrix
            long[][] A1 = new long[k-1][k-1];      // lower right principal submatrix of A, after A[i][i]
            long[][] B  = new long[k-1][k-1];      // matrix containing the powers of A1
            long[][] Bnext = new long[k-1][k-1];   // matrix containing the next power of A1
            long[][] T  = new long[k+1][k];        // the appropriate Toeplitz matrix

            // extract A1 from A
            for (int p=i+1; p<=n-1; p++)
                for (int q=i+1; q<=n-1; q++)
                    A1[p-i-1][q-i-1] = A[p][q];
            // Graph.copy(A1, A, i+1, i+1, n-1, n-1);

            // set B to the unit matrix initially
            for (int p=0; p<k-1; p++)
                B[p][p]=1;
            
            // construct the Toeplitz matrix
            // the upper right part are all zeros
            for (int p=0; p<=k-2; p++)
                for (int q=p+1; q<=k-1; q++)
                    T[p][q]=0;
            
            // the main diagonal are ones
            for (int p=0; p<=k-1; p++)
                T[p][p]=1;
                    
            // the first subdiagonal is -A[i][i]
            for (int p=1; p<=k; p++)
                T[p][p-1]=-A[i][i];
                
            // the following subdiagonals contain the products -RBC
            for (int p=2; p<=k; p++) {
                // first calculate the product -RBC directly
                long r = 0;
                for (int s=0; s<=k-2; s++)
                    for (int t=0; t<=k-2; t++)
                        r += -A[i][i+1+s]*B[s][t]*A[i+1+t][i];
                
                // then set the subdiagonal entries to it
                for (int q=p; q<=k; q++)
                    T[q][q-p]=r;
                
                // then let B be equal to the next power of A1
                Graph.mult(A1, B, Bnext);
                Graph.copy(B, Bnext, 0, 0, k-2, k-2);
            }
            
            // multiply this Toeplitz matrix and v to get a new value for v
            long[][] w = new long[k+1][1];
            Graph.mult(T, v, w);
            v=w;
        }
        
        // At the end, copy the one-column matrix v to a vector and return it
        long[] z = new long[n+1];
        for (int j=0; j<=n; j++)
            z[j] = v[j][0];
        return z;        
    }

    /**
     * Iterative implementation of Berkowitz's algorithm
     * for computing coefficients of the characteristic polynomial of a double matrix.
     */
    public static double[] charpoly(double[][] A) {
        int n = A.length;               // dimension of A
        int i;                          // shows which diagonal entry of A is currently being processed

        // was this an empty matrix?
        if (n==0) {
            double[] z = new double[1];
            z[0]=1.0;
            return z;
        }

        // done separately for n-1
        double[][] v = new double[2][1];
        v[0][0] = 1.0;
        v[1][0] = -A[n-1][n-1];
        
        // going backwards from n-2 to 0
        for (i=n-2; i>=0; i--) {
            int k = n-i;                         // dimension of currently processed submatrix
            double[][] A1 = new double[k-1][k-1];      // lower right principal submatrix of A, after A[i][i]
            double[][] B  = new double[k-1][k-1];      // matrix containing the powers of A1
            double[][] Bnext = new double[k-1][k-1];   // matrix containing the next power of A1
            double[][] T  = new double[k+1][k];        // the appropriate Toeplitz matrix

            // extract A1 from A
            Graph.copy(A1, A, i+1, i+1, n-1, n-1);
            
            // set B to the unit matrix initially
            for (int p=0; p<k-1; p++)
                B[p][p]=1.0;
            
            // construct the Toeplitz matrix
            // the upper right part are all zeros
            for (int p=0; p<=k-2; p++)
                for (int q=p+1; q<=k-1; q++)
                    T[p][q]=0.0;
            
            // the main diagonal are ones
            for (int p=0; p<=k-1; p++)
                T[p][p]=1.0;
                    
            // the first subdiagonal is -A[i][i]
            for (int p=1; p<=k; p++)
                T[p][p-1]=-A[i][i];
                
            // the following subdiagonals contain the products -RBC
            for (int p=2; p<=k; p++) {
                // first calculate the product -RBC directly
                double r = 0;
                for (int s=0; s<=k-2; s++)
                    for (int t=0; t<=k-2; t++)
                        r += -A[i][i+1+s]*B[s][t]*A[i+1+t][i];
                
                // then set the subdiagonal entries to it
                for (int q=p; q<=k; q++)
                    T[q][q-p]=r;
                
                // then let B be equal to the next power of A1
                Graph.mult(A1, B, Bnext);
                Graph.copy(B, Bnext, 0, 0, k-2, k-2);
            }
            
            // multiply this Toeplitz matrix and v to get a new value for v
            double[][] w = new double[k+1][1];
            Graph.mult(T, v, w);
            v=w;
        }
        
        // At the end, copy the one-column matrix v to a vector and return it
        double[] z = new double[n+1];
        for (int j=0; j<=n; j++)
            z[j] = v[j][0];
        return z;        
    }
    
    /**
     * Auxiliary methods to compute C = A*B.
     * Assumes that the appropriate dimensions of A and B match, 
     * and that C has sufficient dimension to hold the product of A and B in its upper left part.
     */
    private static void mult(long[][] A, long[][] B, long[][] C) {
        int p = A.length;
        int q = A[0].length;
        int r = B[0].length;
        long s;
        
        for (int i=0; i<p; i++)
            for (int j=0; j<r; j++) {
                s = 0;
                for (int k=0; k<q; k++)
                    s += A[i][k]*B[k][j];
                C[i][j] = s;
            }
    }
    
    private static void mult(double[][] A, double[][] B, double[][] C) {
        int p = A.length;
        int q = A[0].length;
        int r = B[0].length;
        double s;
        
        for (int i=0; i<p; i++)
            for (int j=0; j<r; j++) {
                s = 0.0;
                for (int k=0; k<q; k++)
                    s += A[i][k]*B[k][j];
                C[i][j] = s;
            }
    }
    
    /**
     * Auxiliary methods to copy the specified submatrix of C to the upper left part of the matrix A.
     * Assumes that A has the appropriate dimensions.
     */
    private static void copy(long[][] A, long[][] C, int fromRow, int fromCol, int toRow, int toCol) {
        for (int i=fromRow; i<=toRow; i++)
            System.arraycopy(C[i], fromCol, A[i-fromRow], 0, toCol-fromCol+1);
    }

    private static void copy(double[][] A, double[][] C, int fromRow, int fromCol, int toRow, int toCol) {
        for (int i=fromRow; i<=toRow; i++)
            System.arraycopy(C[i], fromCol, A[i-fromRow], 0, toCol-fromCol+1);
    }    
    
    /**
     *  Auxiliary function to find eigenvalues of an integer matrix.
     *  "static" means it is a method of the class itself,
     *  so that it has to be called as Graph.spectrum(matrix).
     *  Assumption: mat is a square matrix
     */
    public static double[] spectrum(int[][] mat) {
        int dim = mat.length;
        double[][] dmat = new double[dim][dim];
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                dmat[i][j] = (double) mat[i][j];
            
        return Graph.spectrum(dmat);
    }

    /** 
     * Auxiliary function to find eigenvalues of a double matrix.
     * "static" means it is a method of the class itself,
     * so that it has to be called as Graph.spectrum(matrix).
     * Sorts the eigenvalues in non-decreasing order.
     * Assumption: mat is a square matrix
     */
    public static double[] spectrum(double[][] dmat) {
        int dim = dmat.length;
        DMatrixRMaj ejmlMat = new DMatrixRMaj(dmat);
        EigenDecomposition_F64<DMatrixRMaj> eig = DecompositionFactory_DDRM.eig(dim, false, true);
        if (!eig.decompose(ejmlMat))
            throw new RuntimeException("ejml: computing eigenvalues failed...");
        DMatrixRMaj eL = EigenOps_DDRM.createMatrixD(eig);
        
        double[] eigenvalues = new double[dim];
        for (int i=0; i<dim; i++)
            eigenvalues[i] = eL.getData()[i*dim+i];
        Arrays.sort(eigenvalues);
            
        return eigenvalues;
    }    
    
    /** 
     * Auxiliary function to find eigenvectors of an integer matrix.
     * Eigenvectors are returned in columns.
     * Assumption: mat is a square matrix
     */
    public static double[][] eigenvectors(int[][] mat) {
        int dim = mat.length;
        double[][] dmat = new double[dim][dim];
        for (int i=0; i<dim; i++)
            for (int j=0; j<dim; j++)
                dmat[i][j] = (double) mat[i][j];
            
        return Graph.eigenvectors(dmat);
    }

    /** 
     * Auxiliary function to find eigenvectors of a double matrix.
     * Eigenvectors are returned in columns.
     * Assumption: mat is a square matrix
     * NOTE: eigenvectors do correspond to eigenvalues as returned by spectrum(...).
     */
    public static double[][] eigenvectors(double[][] dmat) {
        // EJML version
        // obtain the eigenvectors
        int dim = dmat.length;
        DMatrixRMaj ejmlMat = new DMatrixRMaj(dmat);
        EigenDecomposition_F64<DMatrixRMaj> eig = DecompositionFactory_DDRM.eig(dim, true, true);
        if (!eig.decompose(ejmlMat))
            throw new RuntimeException("ejml: computing eigenvalues failed...");
        DMatrixRMaj eV = EigenOps_DDRM.createMatrixV(eig);
        double[] eigvecdata = eV.getData();
        
        // sort the eigenvalues while keeping the original indices
        DMatrixRMaj eD = EigenOps_DDRM.createMatrixD(eig);
        double[] eigvaldata = eD.getData();
        class DoubleWithIndices {
            public double value;
            public int index;
        }
        DoubleWithIndices[] eigvalindexed = new DoubleWithIndices[dim];
        for (int i=0; i<dim; i++) {
            eigvalindexed[i] = new DoubleWithIndices();
            eigvalindexed[i].value = eigvaldata[i*dim+i];
            eigvalindexed[i].index = i;
        }
        Arrays.sort(eigvalindexed, new Comparator<DoubleWithIndices>() {
            public int compare(DoubleWithIndices d1, DoubleWithIndices d2) {
                return DoubleUtil.compareTo(d1.value, d2.value);
            }
        });
        
        // pick up eigenvectors so as to follow the sorted eigenvalues
        double[][] eigenvectors = new double[dim][dim];
        for (int col=0; col<dim; col++) {
            int colIndex = eigvalindexed[col].index;
            for (int row=0; row<dim; row++)
                eigenvectors[row][col] = eigvecdata[row*dim+colIndex];
        }
            
        return eigenvectors;
    }    
    
    /** 
     * Adjacency spectrum and eigenvectors
     * Aspectrum()[0] is the smallest, Aspectrum()[n-1] is the largest eigenvalue
     */
    public double[] Aspectrum() {
        return Graph.spectrum(A);
    }
    
    /**
     * Checks whether the graph contains nearly 0 in its adjacency spectrum.
     */
    public boolean Asingular() {
        double[] eig = Aspectrum();
        for (int i=0; i<n; i++)
            if (DoubleUtil.equals(eig[i], 0.0))
                return true;
        return false;
    }
    
    public double[][] Aeigenvectors() {
        return Graph.eigenvectors(A);
    }
    
    /** 
     * Laplacian spectrum and eigenvectors
     * Lspectrum()[0] is the smallest, Lspectrum()[n-1] is the largest eigenvalue
     */
    public double[] Lspectrum() {
        Lmatrix();
        return Graph.spectrum(L);
    }
    
    public double[][] Leigenvectors() {
        Lmatrix();
        return Graph.eigenvectors(L);
    }
    
    public double[] fiedlerVector() {
        return Graph.extractColumn(Leigenvectors(), 1);
    }
    
    /**
     * Signless Laplacian spectrum and eigenvectors
     * Qspectrum()[0] is the smallest, Qspectrum()[n-1] is the largest eigenvalue
     */
    public double[] Qspectrum() {
        Qmatrix();
        return Graph.spectrum(Q);
    }
    
    public double[][] Qeigenvectors() {
        Qmatrix();
        return Graph.eigenvectors(Q);
    }
        
    /** 
     * Distance spectrum and eigenvectors
     * Dspectrum()[0] is the smallest, Dspectrum()[n-1] is the largest eigenvalue
     */
    public double[] Dspectrum() {
        Dmatrix();
        return Graph.spectrum(D);
    }

    public double[][] Deigenvectors() {
        Dmatrix();
        return Graph.eigenvectors(D);
    }
    
    /**
     * Distance Laplacian spectrum and eigenvectors
     */
    public double[] DLspectrum() {
        DLmatrix();
        return Graph.spectrum(DL);
    }
    
    public double[][] DLeigenvectors() {
        DLmatrix();
        return Graph.eigenvectors(DL);
    }
    
    /** 
     * Modularity spectrum
     * Mspectrum()[0] is the smallest, Mspectrum()[n-1] is the largest eigenvalue
     */
    public double[] Mspectrum() {
        Mmatrix();
        return Graph.spectrum(M);
    }
    
    public double[][] Meigenvectors() {
        Mmatrix();
        return Graph.eigenvectors(M);
    }
    
    /** 
     * Checks whether two graphs have the same adjacency spectrum
     */
    public boolean Acospectral(Graph h) {
        return DoubleUtil.equals(Aspectrum(), h.Aspectrum());
    }
    
    /**
     * Checks whether two graphs have the same Laplacian spectrum
     */
    public boolean Lcospectral(Graph h) {
        return DoubleUtil.equals(Lspectrum(), h.Lspectrum());
    }
    
    /** 
     * Checks whether two graphs have the same signless Laplacian spectrum
     */
    public boolean Qcospectral(Graph h) {
        return DoubleUtil.equals(Qspectrum(), h.Qspectrum());
    }
    
    /** 
     * Checks whether two graphs have the same distance spectrum
     */
    public boolean Dcospectral(Graph h) {
        return DoubleUtil.equals(Dspectrum(), h.Dspectrum());
    }
    
    /**
     * Checks whether two graphs have the same distance Laplacian spectrum
     */
    public boolean DLcospectral(Graph h) {
        return DoubleUtil.equals(DLspectrum(), h.DLspectrum());
    }
    
    /** 
     * Checks whether two graphs have the same modularity spectrum
     */
    public boolean Mcospectral(Graph h) {
        return DoubleUtil.equals(Mspectrum(), h.Mspectrum());
    }

    /** 
     * Auxiliary function to check whether an integer matrix has integer eigenvalues.
     * "static" means it is a method of the class itself,
     * so that it has to be called as Graph.integralSpectrum(matrix).
     * Assumption: mat is a square matrix
     */
    public static boolean integralSpectrum(int[][] mat) {
        double[] eigenvalues = Graph.spectrum(mat);
        int dim = mat.length;
        
        for (int i=0; i<dim; i++)
            if (!DoubleUtil.equals(eigenvalues[i], (double) Math.round(eigenvalues[i])))
                return false;
        
        return true;
    }    

    /**
     * Auxiliary function to check whether a double matrix has integer eigenvalues.
     * Assumption: mat is a square matrix
     */
    public static boolean integralSpectrum(double[][] dmat) {
        double[] eigenvalues = Graph.spectrum(dmat);
        int dim = dmat.length;
        
        for (int i=0; i<dim; i++)
            if (!DoubleUtil.equals(eigenvalues[i], (double) Math.round(eigenvalues[i])))
                return false;
        
        return true;
    }    

    /** 
     * Is adjacency spectrum integral?
     */
    public boolean Aintegral() {
        return Graph.integralSpectrum(A);
    }
    
    /** 
     * Is Laplacian spectrum integral?
     */
    public boolean Lintegral() {
        Lmatrix();
        return Graph.integralSpectrum(L);
    }
    
    /** 
     * Is signless Laplacian spectrum integral?
     */
    public boolean Qintegral() {
        Qmatrix();
        return Graph.integralSpectrum(Q);
    }
    
    /** 
     * Is distance spectrum integral?
     */
    public boolean Dintegral() {
        Dmatrix();
        return Graph.integralSpectrum(D);
    }
    
    /**
     * Is distance Laplacian spectrum integral?
     */
    public boolean DLintegral() {
        DLmatrix();
        return Graph.integralSpectrum(DL);
    }
    
    /**
     * Is modularity spectrum integral?
     */
    public boolean Mintegral() {
        Mmatrix();
        return Graph.integralSpectrum(M);
    }
        
    /**
     * Auxiliary function for calculation of energies
     */
    public static double deviation(double[] eigs) {
        double average = 0.0;
        for (int i=0; i < eigs.length; i++)
            average += eigs[i];
        average = average / eigs.length;
        
        double deviation = 0.0;
        for (int i=0; i< eigs.length; i++)
            deviation += Math.abs(eigs[i] - average);
            
        return deviation;
    }
    
    /** 
     * Matrix energy is deviation from the average of its eigenvalues.
     * "static" means it is a method of the class itself,
     * so that it has to be called as Graph.matrixEnergy(matrix).
     * Assumption: mat is a square matrix
     */
    public static double matrixEnergy(int[][] mat) {
        return Graph.deviation(Graph.spectrum(mat));
    }

    public static double matrixEnergy(double[][] dmat) {
        return Graph.deviation(Graph.spectrum(dmat));
    }

    /**
     * Adjacency energy
     */
    public double energy() {
        return Aenergy();
    }
    
    public double Aenergy() {
        return Graph.deviation(Aspectrum());
    }
    
    /**
     * Laplacian energy
     */
    public double Lenergy() {
        return Graph.deviation(Lspectrum());
    }
    
    /**
     * Signless Laplacian energy
     */
    public double Qenergy() {
        return Graph.deviation(Qspectrum());
    }
    
    /**
     * Distance energy
     */
    public double Denergy() {
        return Graph.deviation(Dspectrum());
    }
    
    /**
     * Distance Laplacian energy
     */
    public double DLenergy() {
        return Graph.deviation(DLspectrum());
    }
    
    /**
     * Modularity energy
     */
    public double Menergy() {
        return Graph.deviation(Mspectrum());
    }
    
    /** 
     * LEL, Laplacian-like energy
     */
    public double LEL() {
        double[] eigs = Lspectrum();
        
        double lel = 0.0;
        for (int i=0; i<n; i++)
            if (eigs[i]>0)
                lel += Math.sqrt(eigs[i]);
                
        return lel;
    }
    
    /**
     * Estrada index
     */ 
    public double estrada() {
        double[] eigs = Aspectrum();
        
        double estrada = 0.0;
        for (int i=0; i<n; i++)
            estrada += Math.exp(eigs[i]);
        return estrada;
    }
    
    /** 
     * Laplacian Estrada index
     */
    public double Lestrada() {
        double[] eigs = Lspectrum();
        
        double lestrada = 0.0;
        for (int i=0; i<n; i++)
            lestrada += Math.exp(eigs[i]);
        return lestrada;
    }
    
    /**
     * Diameter
     */ 
    public int diameter() {
        Dmatrix();
        int diameter = 0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (D[i][j]>diameter)
                    diameter = D[i][j];
        return diameter;
    }
    
    /** 
     * Radius
     */
    public int radius() {
        Dmatrix();
        int radius = n;
        for (int i=0; i<n; i++) {
            int ecc = 0;
            for (int j=0; j<n; j++)
                if (D[i][j]>ecc)
                    ecc = D[i][j];
            if (ecc<radius)
                radius=ecc;
        }
        return radius;
    }
    
    /** 
     * Wiener index
     */
    public int wiener() {
        Dmatrix();
        int wiener = 0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                wiener += D[i][j];
        return wiener;
    }

    /**
     * Vertex transmissions
     */
    public int[] transmissions() {
        Dmatrix();
        int[] transmission = new int[n];
        for (int i=0; i<n; i++) {
            transmission[i]=0;
            for (int j=0; j<n; j++)
                transmission[i] += D[i][j];
        }
        return transmission;
    }
    
    public boolean transmissionIrregular() {
        int[] t = transmissions();

        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (t[i]==t[j])
                    return false;

        return true;
    }
    
    /**
     * According to Klavzar, a graph G is multiset distance irregular if
     * for every two vertices u and v, the multisets (i.e., families) m(u|V) and m(v|V) are different.
     * Here m(u|V) denotes the family of distances from u to all other vertices in V.
     * In the method, the family {{0,...,0 (m_0 times), 1,...,1 (m_1 times), ..., n-1,...,n-1 (m_{n-1} times)}}
     * is represented as a vector (m_0, m_1, ..., m_{n-1}).
     */
    public int[][] multisetDistances() {
        Dmatrix();

        // create the matrix of multiplicities,
        // where the i-th row contains the above vector of multiplicities for the vertex i
        int[][] mpc = new int[n][n];
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                mpc[i][j]=0;
        
        // now pass through all the distances to properly update the multiplicities
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                mpc[i][D[i][j]]++;
        
        return mpc;        
    }
    
    public boolean multisetDistanceIrregular() {
        int[][] mpc = multisetDistances();
        
        // check whether all the rows in mpc are different
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (equalRows(mpc, i, j))
                    return false;
                    
        return true;
    }

    // auxiliary method that check whether the rows i and j of the integer matrix mpc are equal or not
    public boolean equalRows(int[][] mpc, int i, int j) {
        int l = mpc[0].length;
        
        for (int k=0; k<l; k++)
            if (mpc[i][k]!=mpc[j][k])
                return false;
                
        return true;
    }
    
    /**
     * Soltes' sum
     */
    public int soltesSum() {
        int ss = 0;
        int ww = wiener();
        
        for (int i=0; i<n; i++) {
            Graph h = deleteVertices(new int[]{i});
            ss += Math.abs(ww - h.wiener());
        }
        
        return ss;
    }
    
    /** 
     * Szeged index
     */
    public int szeged() {
        Dmatrix();
        int szeged = 0;
        for (int u=0; u<n; u++)
            for (int v=u+1; v<n; v++)
                if (A[u][v]==1) {
                    int nu=0;
                    int nv=0;
                    for (int k=0; k<n; k++) {
                        if (D[k][u]<D[k][v]) 
                            nu++;
                        if (D[k][v]<D[k][u])
                            nv++;
                    }
                    szeged += nu*nv;
                }
        return szeged;
    }    

    /** 
     * Weighted Szeged index
     */
    public int weightedSzeged() {
        Dmatrix();
        int wszeged = 0;
        for (int u=0; u<n; u++)
            for (int v=u+1; v<n; v++)
                if (A[u][v]==1) {
                    int nu=0;
                    int nv=0;
                    for (int k=0; k<n; k++) {
                        if (D[k][u]<D[k][v]) 
                            nu++;
                        if (D[k][v]<D[k][u])
                            nv++;
                    }
                    wszeged += (degree[u]+degree[v])*nu*nv;
                }
        return wszeged;
    }    
    
    /** 
     * Randic index
     */
    public double randic() {
        double randic = 0.0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    randic += 1/Math.sqrt(degree[i]*degree[j]);
        return randic;
    }    
    
    /** 
     * First Zagreb index
     */
    public int zagreb1() {
        int zagreb1 = 0;
        for (int i=0; i<n; i++)
            zagreb1 += degree[i]*degree[i];
        return zagreb1;
    }
    
    /**
     * Second Zagreb index
     */
    public int zagreb2() {
        int zagreb2 = 0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    zagreb2 += degree[i]*degree[j];
        return zagreb2;
    }
    
    public double sombor() {
        double sombor=0.0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    sombor += Math.sqrt(degree[i]*degree[i] + degree[j]*degree[j]);
        return sombor;
    }
    
    /**
     * Arithmetic-geometric index
     */
    public double ag() {
        double ag = 0.0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    ag += (degree[i]+degree[j])/(2.0*Math.sqrt(degree[i]*degree[j]));
        return ag;
    }
    
    /** 
     * Geometric-arithmetic index
     */
    public double ga() {
        double ga = 0.0;
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    ga += (2.0*Math.sqrt(degree[i]*degree[j]))/(degree[i]+degree[j]);
        return ga;
    }
    
    /**
     * Computes the expression from AG-GA1 conjecture from
     * S. Vujosevíc, G. Popivoda, Z. Kovijaníc-Vukícevíc, B. Furtula, R. Skrekovski, 
     * Arithmetic-geometric index and its relations with geometric-arithmetic index, 
     * Appl. Math. Comput. 391 (2021), 125706.
     */
    public double skrek_popivoda() {
        double c = 0.116515138991168;       // = (sqrt(21)-4)/5
        
        double r1 = Math.floor(c*n);
        double f1 = r1*(n-r1)*(n-r1-1)*(n-r1-1) / (2.0*(n+r1-1)*Math.sqrt((n-1)*r1));
        
        double r2 = Math.ceil(c*n);
        double f2 = r2*(n-r2)*(n-r2-1)*(n-r2-1) / (2.0*(n+r2-1)*Math.sqrt((n-1)*r2));
        
        return ag() - ga() - Math.max(f1, f2);
    }
    
    /** 
     * Distance-sum heterogeneity index is defined by Estrada and Vargas-Estrada
     * in Appl. Math. Comput. 218 (2012), 10393-10405 as
     * dshi = \sum_{i=1}^n \frac{d_i}{s_i} - 2\sum_{ij\in E} (s_is_j)^{-1/2},
     * where d_i is the degree of vertex i, while s_i is the sum of distances from i to all other vertices.
     */
    public double dshi() {
        Dmatrix();
        int[] s = new int[n];
        for (int i=0; i<n; i++) {
            s[i]=0;
            for (int j=0; j<n; j++)
                s[i] += D[i][j];
        }
    
        double dshi = 0.0;
        for (int i=0; i<n; i++)
            dshi += ((double)degree[i])/s[i];
            
        for (int i=0; i<n; i++)
            for (int j=0; j<i; j++)
                if (A[i][j]==1)
                    dshi -= 2/Math.sqrt(s[i]*s[j]);

        return dshi;
    }

    /**
     * Auxiliary methods to count numbers of small cliques and independent sets in a graph,
     * used to test capability of reinforcement learning to construct Ramsey graphs.
     */
    public long num3cliques() {
        long cliques = 0;
        for (int a=0; a<n-2; a++)
            for (int b=a+1; b<n-1; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            cliques++;
        return cliques;
    }
    
    public long num4cliques() {
        long cliques = 0;
        for (int a=0; a<n-3; a++)
            for (int b=a+1; b<n-2; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n-1; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==1)&&(A[b][d]==1)&&(A[c][d]==1))
                                    cliques++;
        return cliques;
    }
    
    public long num5cliques() {
        long cliques = 0;
        for (int a=0; a<n-4; a++)
            for (int b=a+1; b<n-3; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n-2; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            for (int d=c+1; d<n-1; d++)
                                if ((A[a][d]==1)&&(A[b][d]==1)&&(A[c][d]==1))
                                    for (int e=d+1; e<n; e++)
                                        if ((A[a][e]==1)&&(A[b][e]==1)&&(A[c][e]==1)&&(A[d][e]==1))
                                            cliques++;
        return cliques;
    }

    public long num6cliques() {
        long cliques = 0;
        for (int a=0; a<n-5; a++)
            for (int b=a+1; b<n-4; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n-3; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            for (int d=c+1; d<n-2; d++)
                                if ((A[a][d]==1)&&(A[b][d]==1)&&(A[c][d]==1))
                                    for (int e=d+1; e<n-1; e++)
                                        if ((A[a][e]==1)&&(A[b][e]==1)&&(A[c][e]==1)&&(A[d][e]==1))
                                            for (int f=e+1; f<n; f++)
                                                if ((A[a][f]==1)&&(A[b][f]==1)&&(A[c][f]==1)
                                                  &&(A[d][f]==1)&&(A[e][f]==1))
                                                    cliques++;
        return cliques;
    }
    
    public long num7cliques() {
        long cliques = 0;
        for (int a=0; a<n-6; a++)
            for (int b=a+1; b<n-5; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n-4; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            for (int d=c+1; d<n-3; d++)
                                if ((A[a][d]==1)&&(A[b][d]==1)&&(A[c][d]==1))
                                    for (int e=d+1; e<n-2; e++)
                                        if ((A[a][e]==1)&&(A[b][e]==1)&&(A[c][e]==1)&&(A[d][e]==1))
                                            for (int f=e+1; f<n-1; f++)
                                                if ((A[a][f]==1)&&(A[b][f]==1)&&(A[c][f]==1)
                                                  &&(A[d][f]==1)&&(A[e][f]==1))
                                                    for (int g=f+1; g<n; g++)
                                                        if ((A[a][g]==1)&&(A[b][g]==1)&&(A[c][g]==1)
                                                          &&(A[d][g]==1)&&(A[e][g]==1)&&(A[f][g]==1))
                                                            cliques++;
        return cliques;
    }
    
    public long num8cliques() {
        long cliques = 0;
        for (int a=0; a<n-7; a++)
            for (int b=a+1; b<n-6; b++)
                if (A[a][b]==1)
                    for (int c=b+1; c<n-5; c++)
                        if ((A[a][c]==1)&&(A[b][c]==1))
                            for (int d=c+1; d<n-4; d++)
                                if ((A[a][d]==1)&&(A[b][d]==1)&&(A[c][d]==1))
                                    for (int e=d+1; e<n-3; e++)
                                        if ((A[a][e]==1)&&(A[b][e]==1)&&(A[c][e]==1)&&(A[d][e]==1))
                                            for (int f=e+1; f<n-2; f++)
                                                if ((A[a][f]==1)&&(A[b][f]==1)&&(A[c][f]==1)
                                                  &&(A[d][f]==1)&&(A[e][f]==1))
                                                    for (int g=f+1; g<n-1; g++)
                                                        if ((A[a][g]==1)&&(A[b][g]==1)&&(A[c][g]==1)
                                                          &&(A[d][g]==1)&&(A[e][g]==1)&&(A[f][g]==1))
                                                            for (int h=g+1; h<n; h++)
                                                                if ((A[a][h]==1)&&(A[b][h]==1)&&(A[c][h]==1)&&(A[d][h]==1)
                                                                    &&(A[e][h]==1)&&(A[f][h]==1)&&(A[g][h]==1))
                                                                    cliques++;
        return cliques;
    }

    public long num3cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-2; a++)
            for (int b=a+1; b<n-1; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            cocliques++;
        return cocliques;
    }
    
    public long num4cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-3; a++)
            for (int b=a+1; b<n-2; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n-1; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==0)&&(A[b][d]==0)&&(A[c][d]==0))
                                    cocliques++;
        return cocliques;
    }
    
    public long num5cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-4; a++)
            for (int b=a+1; b<n-3; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n-2; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            for (int d=c+1; d<n-1; d++)
                                if ((A[a][d]==0)&&(A[b][d]==0)&&(A[c][d]==0))
                                    for (int e=d+1; e<n; e++)
                                        if ((A[a][e]==0)&&(A[b][e]==0)&&(A[c][e]==0)&&(A[d][e]==0))
                                            cocliques++;
        return cocliques;
    }

    public long num6cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-5; a++)
            for (int b=a+1; b<n-4; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n-3; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            for (int d=c+1; d<n-2; d++)
                                if ((A[a][d]==0)&&(A[b][d]==0)&&(A[c][d]==0))
                                    for (int e=d+1; e<n-1; e++)
                                        if ((A[a][e]==0)&&(A[b][e]==0)&&(A[c][e]==0)&&(A[d][e]==0))
                                            for (int f=e+1; f<n; f++)
                                                if ((A[a][f]==0)&&(A[b][f]==0)&&(A[c][f]==0)
                                                  &&(A[d][f]==0)&&(A[e][f]==0))
                                                    cocliques++;
        return cocliques;
    }
    
    public long num7cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-6; a++)
            for (int b=a+1; b<n-5; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n-4; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            for (int d=c+1; d<n-3; d++)
                                if ((A[a][d]==0)&&(A[b][d]==0)&&(A[c][d]==0))
                                    for (int e=d+1; e<n-2; e++)
                                        if ((A[a][e]==0)&&(A[b][e]==0)&&(A[c][e]==0)&&(A[d][e]==0))
                                            for (int f=e+1; f<n-1; f++)
                                                if ((A[a][f]==0)&&(A[b][f]==0)&&(A[c][f]==0)
                                                  &&(A[d][f]==0)&&(A[e][f]==0))
                                                    for (int g=f+1; g<n; g++)
                                                        if ((A[a][g]==0)&&(A[b][g]==0)&&(A[c][g]==0)
                                                          &&(A[d][g]==0)&&(A[e][g]==0)&&(A[f][g]==0))
                                                            cocliques++;
        return cocliques;
    }
    
    public long num8cocliques() {
        long cocliques = 0;
        for (int a=0; a<n-7; a++)
            for (int b=a+1; b<n-6; b++)
                if (A[a][b]==0)
                    for (int c=b+1; c<n-5; c++)
                        if ((A[a][c]==0)&&(A[b][c]==0))
                            for (int d=c+1; d<n-4; d++)
                                if ((A[a][d]==0)&&(A[b][d]==0)&&(A[c][d]==0))
                                    for (int e=d+1; e<n-3; e++)
                                        if ((A[a][e]==0)&&(A[b][e]==0)&&(A[c][e]==0)&&(A[d][e]==0))
                                            for (int f=e+1; f<n-2; f++)
                                                if ((A[a][f]==0)&&(A[b][f]==0)&&(A[c][f]==0)
                                                  &&(A[d][f]==0)&&(A[e][f]==0))
                                                    for (int g=f+1; g<n-1; g++)
                                                        if ((A[a][g]==0)&&(A[b][g]==0)&&(A[c][g]==0)
                                                          &&(A[d][g]==0)&&(A[e][g]==0)&&(A[f][g]==0))
                                                            for (int h=g+1; h<n; h++)
                                                                if ((A[a][h]==0)&&(A[b][h]==0)&&(A[c][h]==0)&&(A[d][h]==0)
                                                                  &&(A[e][h]==0)&&(A[f][h]==0)&&(A[g][h]==0))
                                                                    cocliques++;
        return cocliques;
    }
    
    // jgrapht translations
    
    /**
     * Returns jgrapht's SimpleGraph representation of a constructed graph,
     * which is needed for using jgrapht's methods.
     */
    public SimpleGraph<Integer, DefaultEdge> toJGraphT() {
        SimpleGraph<Integer, DefaultEdge> g = new SimpleGraph<>(DefaultEdge.class);
        Integer[] v = new Integer[n];
        for (int i=0; i<n; i++) {
            v[i] = Integer.valueOf(i);
            g.addVertex(v[i]);
        }
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (A[i][j]==1)
                    g.addEdge(v[i],v[j]);
        return g;
    }

    /**
     * Tests whether this and another graph are isomorphic
     * and returns the appropriate boolean value.
     */
    public boolean isIsomorphic(Graph other) {
        VF2GraphIsomorphismInspector<Integer, DefaultEdge> isoThisOther
                = new VF2GraphIsomorphismInspector<>(this.toJGraphT(), other.toJGraphT());
        return isoThisOther.isomorphismExists();
    }

    /**
     * Returns the cardinality of the maximum matching.
     */
    public int matchingNumber() {
        DenseEdmondsMaximumCardinalityMatching demcm = new DenseEdmondsMaximumCardinalityMatching(this.toJGraphT());
        MatchingAlgorithm.Matching<Integer, DefaultEdge> ma = demcm.getMatching();
        return ma.getEdges().size();
    }
    
    // auxiliary spectral functions
    
    /**
     * Eigenvectors are placed in columns, so we need 
     * an auxiliary function to extract a column from a matrix.
     */
    public static int[] extractColumn(int[][] mat, int column) {
        int[] excol = new int[mat.length];
        
        for (int i=0; i<mat.length; i++)
            excol[i] = mat[i][column];
            
        return excol;
    }

    public static double[] extractColumn(double[][] mat, int column) {
        double[] excol = new double[mat.length];
        
        for (int i=0; i<mat.length; i++)
            excol[i] = mat[i][column];
            
        return excol;
    }

    /**
     * This method returns a new Graph object that represents
     * a vertex-deleted subgraph of this graph
     */
    public Graph deleteVertices(int[] verticesToDelete) {
        // indicate which vertices will remain in the subgraph
        int[] verticesToKeep = new int[n];
        for (int i=0; i<n; i++)
            verticesToKeep[i]=1;
        for (int j=0; j<verticesToDelete.length; j++)
            verticesToKeep[verticesToDelete[j]]=0;

        // so, how many vertices remain in the subgraph?
        int n1=0;
        for (int i=0; i<n; i++)
            if (verticesToKeep[i]==1)
                n1++;

        // the array of vertices that remain and the adjacency matrix of such induced subgraph
        int[] remainingVertices = new int[n1];
        int currentIndex = 0;
        for (int i=0; i<n; i++)
            if (verticesToKeep[i]==1) {
                remainingVertices[currentIndex]=i;
                currentIndex++;
            }

        int[][] A1 = new int[n1][n1];
        for (int i=0; i<n1; i++)
            for (int j=0; j<n1; j++)
                A1[i][j] = A[remainingVertices[i]][remainingVertices[j]];

        // return the induced subgraph determined by the submatrix A1
        return new Graph(A1);
    }

    /** 
     * Output printing formats for graph, its vectors and its matrices
     * Returns multiline string representing integer matrix
     * delims is a three-character string, 
     * where character at position 0 is put at the beginning of a matrix,
     * character at position 1 is put between entries,
     * and character at position 2 is put at the end of a matrix (think of "[,]").
     */
    public static String printVector(int[] vec, String delims) {
        StringBuffer buf = new StringBuffer("");
        
        buf.append(delims.charAt(0));
        for (int i=0; i<vec.length; i++) {
            buf.append("" + vec[i]);
            if (i!=vec.length-1)    // was it the last entry?
                buf.append(delims.charAt(1) + " ");
        }
        buf.append(delims.charAt(2));
        
        return buf.toString();
    }

    public static String printVector(int[] vec) {
        return Graph.printVector(vec, "[,]");
    }
    
    public static String printMatrix(int[][] mat, String delims) {
        StringBuffer buf = new StringBuffer("");

        buf.append(delims.charAt(0));
        for (int i=0; i<mat.length; i++) {
            buf.append(delims.charAt(0));
            for (int j=0; j<mat[i].length; j++) {
                buf.append("" + mat[i][j]);
                
                if (j!=mat[i].length-1)         // was it the last column?
                    buf.append(delims.charAt(1) + " ");
            }
            buf.append(delims.charAt(2));

            if (i!=mat.length-1)                // was it the last row?
                buf.append(delims.charAt(1) + " ");
        }
        buf.append(delims.charAt(2));
        
        return buf.toString();
    }

    public static String printMatrix(int[][] mat) {
        return Graph.printMatrix(mat, "[,]");
    }
    
    /** 
     * Returns multiline string representing double vector or double matrix
     */
    public static String printVector(double[] vec, String delims) {
        StringBuffer buf = new StringBuffer("");
        
        buf.append(delims.charAt(0));
        for (int i=0; i<vec.length; i++) {
            buf.append("" + vec[i]);
            if (i!=vec.length-1)    // was it the last entry?
                buf.append(delims.charAt(1) + " ");
        }
        buf.append(delims.charAt(2));
        
        return buf.toString();
    }

    public static String printVector(double[] vec) {
        return Graph.printVector(vec, "[,]");
    }

    public static String printMatrix(double[][] dmat, String delims) {
        StringBuffer buf = new StringBuffer("");

        buf.append(delims.charAt(0));
        for (int i=0; i<dmat.length; i++) {
            buf.append(delims.charAt(0));
            for (int j=0; j<dmat[i].length; j++) {
                buf.append("" + dmat[i][j]);
                
                if (j!=dmat[i].length-1)         // was it the last column?
                    buf.append(delims.charAt(1) + " ");
            }
            buf.append(delims.charAt(2));

            if (i!=dmat.length-1)               // was it the last row?
                buf.append(delims.charAt(1) + " ");
        }
        buf.append(delims.charAt(2));
        
        return buf.toString();
    }

    public static String printMatrix(double[][] dmat) {
        return Graph.printMatrix(dmat, "[,]");
    }
    
    /** 
     * String representing adjacency matrix
     */
    public String printAmatrix() {
        return Graph.printMatrix(A);
    }

    /**
     * String representing Laplacian matrix    
     */
    public String printLmatrix() {
        Lmatrix();
        return Graph.printMatrix(L);
    }
    
    /**
     * String representing signless Laplacian matrix
     */
    public String printQmatrix() {
        Qmatrix();
        return Graph.printMatrix(Q);
    }
    
    /**
     * String representing distance matrix
     */
    public String printDmatrix() {
        Dmatrix();
        return Graph.printMatrix(D);
    }
    
    /**
     * String representing modularity matrix
     */
    public String printMmatrix() {
        Mmatrix();
        return Graph.printMatrix(M);
    }
    
    /**
     * Returns string containing list of edges
     */
    public String printEdgeList() {
        StringBuffer buf = null;

        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (A[i][j]==1) {
                    if (buf == null) {
                        buf = new StringBuffer("");
                    }
                    else
                        buf.append(", ");
                    buf.append("" + i + " " + j);
                }
        
        return buf.toString();
    }
    
    /**
     * Returns string describing graph in a .dot format,
     * needed for visualisation with Graphviz.
     */
    public String printDotFormat() {
        StringBuffer buf = new StringBuffer("Graph {\n");
        
        for (int i=0; i<n; i++)
            buf.append("" + i + " [shape=circle]\n");
            
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (A[i][j]==1)
                    buf.append("" + i + " -- " + j + "\n");
        
        buf.append("}\n");
        return buf.toString();
    }
        
    /**
     * Together with the graph, you can visualise additional data
     * by placing them in a data string. 
     * This string is put as a label of a separate isolated vertex,
     * and visualised by Graphviz in the same image next to the graph itself.
     */
    public String printDotFormat(String data) {
        StringBuffer buf = new StringBuffer("Graph {\n");

        for (int i=0; i<n; i++)
            buf.append("" + i + " [shape=circle]\n");
            
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (A[i][j]==1)
                    buf.append("" + i + " -- " + j + "\n");
        
        buf.append("data [shape=box, label=\"" + data + "\"]\n");            
        buf.append("}\n");
        return buf.toString();        
    }
    
    /** Together with the graph, you can visualise additional data associated with each vertex,
     *  by placing them in an array of strings 
     *  (assuming here that the number of strings is equal to the number of vertices!).
     */
    public String printDotFormat(String[] data) {
        StringBuffer buf = new StringBuffer("Graph {\n");

        for (int i=0; i<n; i++)
            buf.append("" + i + " [shape=box, label=\"" + data[i] + "\"]\n");
            
        for (int i=0; i<n; i++)
            for (int j=i+1; j<n; j++)
                if (A[i][j]==1)
                    buf.append("" + i + " -- " + j + "\n");
        
        buf.append("}\n");
        return buf.toString();        
    }
    
    /** 
     * Writes the .dot format description of a graph to the file
     */
    public void saveDotFormat(String filename) throws IOException {
        PrintWriter outfile = new PrintWriter(new BufferedWriter(new FileWriter(filename)));
        outfile.println(printDotFormat());
        outfile.close();
    }
    
    /**
     * Writes the .dot format description of a graph to the file,
     * together with additional data placed as a label of a separate isolated vertex
     */
    public void saveDotFormat(String filename, String data) throws IOException {
        PrintWriter outfile = new PrintWriter(new BufferedWriter(new FileWriter(filename)));
        outfile.println(printDotFormat(data));
        outfile.close();
    }
    
    /**
     * Writes the .dot format description of a graph to the file,
     * together with additional data associated with each vertex
     */
    public void saveDotFormat(String filename, String[] data) throws IOException {
        PrintWriter outfile = new PrintWriter(new BufferedWriter(new FileWriter(filename)));
        outfile.println(printDotFormat(data));
        outfile.close();
    }
}