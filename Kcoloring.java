/**
 * Receives a symmetric matrix whose entries have values 0,...,m-1,
 * which represent edge coloring of Kn with m colors.
 * Implements counters for the numbers of copies of various small graphs
 * in the subgraph of Kn formed by edges with a given color.
 */
package graph6java;

public class Kcoloring
{
    private int n;      // the number of vertices
    private int[][] A;  // the edge coloring of Kn    
    
    public Kcoloring() {
        n = 0;
    }
    
    public Kcoloring(int[][] A) {
        n = A.length;
        this.A = A;
    }

    /**
     * Methods for counting the numbers of copies of various small graphs 
     * in the subgraph of Kn formed by the edges of a given color
     */
    public long numK3(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&(A[b][c]==color))
                            num++;
        return num;
    }
    
    public long numK4(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&(A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&
                                    (A[b][d]==color)&&
                                    (A[c][d]==color))
                                    num++;
        return num;
    }

    public long numK5(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&
                                    (A[b][d]==color)&&
                                    (A[c][d]==color))
                                    for (int e=d+1; e<n; e++)
                                        if ((A[a][e]==color)&&
                                            (A[b][e]==color)&&
                                            (A[c][e]==color)&&
                                            (A[d][e]==color))
                                            num++;
        return num;
    }
    
    public long numK6(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&
                                    (A[b][d]==color)&&
                                    (A[c][d]==color))
                                    for (int e=d+1; e<n; e++)
                                        if ((A[a][e]==color)&&
                                            (A[b][e]==color)&&
                                            (A[c][e]==color)&&
                                            (A[d][e]==color))
                                            for (int f=e+1; f<n; f++)
                                                if ((A[a][f]==color)&&
                                                    (A[b][f]==color)&&
                                                    (A[c][f]==color)&&
                                                    (A[d][f]==color)&&
                                                    (A[e][f]==color))
                                                    num++;
        return num;
    }

    public long numK4e(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                            cand++;
                    num += (cand*(cand-1))/2;
                }
        return num;
    }

    public long numK5e(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&
                            (A[b][c]==color)) {
                                int cand = 0;
                                for (int u=0; u<n; u++)
                                    if ((u!=a)&&(A[u][a]==color)&&
                                        (u!=b)&&(A[u][b]==color)&&
                                        (u!=c)&&(A[u][c]==color))
                                        cand++;
                                num += (cand*(cand-1))/2;
                            }
        return num;
    }
    
    public long numK6e(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&
                            (A[b][c]==color)) 
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&
                                    (A[b][d]==color)&&
                                    (A[c][d]==color)) {
                                        int cand = 0;
                                        for (int u=0; u<n; u++)
                                            if ((u!=a)&&(A[u][a]==color)&&
                                                (u!=b)&&(A[u][b]==color)&&
                                                (u!=c)&&(A[u][c]==color)&&
                                                (u!=d)&&(A[u][d]==color))
                                                cand++;
                                        num += (cand*(cand-1))/2;
                            }
        return num;
    }

    public long numK24(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++) {
                int cand = 0;
                for (int u=0; u<n; u++)
                    if ((u!=a)&&(A[u][a]==color)&&
                        (u!=b)&&(A[u][b]==color))
                            cand++;
                num += (cand*(cand-1)*(cand-2)*(cand-3))/24;
            }
        return num;                
    }
    
    public long numK25(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++) {
                int cand = 0;
                for (int u=0; u<n; u++)
                    if ((u!=a)&&(A[u][a]==color)&&
                        (u!=b)&&(A[u][b]==color))
                            cand++;
                num += (cand*(cand-1)*(cand-2)*(cand-3)*(cand-4))/120;
            }
        return num;                
    }
    
    public long numK33(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                for (int c=b+1; c<n; c++) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color)&&
                            (u!=c)&&(A[u][c]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2))/6;
                }
        return num;
    }
    
    public long numK34(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                for (int c=b+1; c<n; c++) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color)&&
                            (u!=c)&&(A[u][c]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2)*(cand-3))/24;
                }
        return num;
    }
    
    public long numK35(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                for (int c=b+1; c<n; c++) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color)&&
                            (u!=c)&&(A[u][c]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2)*(cand-3)*(cand-4))/120;
                }
        return num;
    }

    public long numK222(int color) {
        long num = 0;
        for (int a=0; a<n; a++)             // a is generally the smallest vertex of K222
            for (int b=a+1; b<n; b++)
                for (int c=a+1; c<n; c++)       // c is smallest among vertices of K222-a-b
                    if ((A[a][c]==color)&&
                        (A[b][c]==color)&&(c!=b))
                        for (int d=c+1; d<n; d++)
                            if ((A[a][d]==color)&&
                                (A[b][d]==color)&&(d!=b)) {
                                    int cand = 0;
                                    for (int u=c+1; u<n; u++)
                                        if ((A[u][a]==color)&&
                                            (A[u][b]==color)&&(u!=b)&&
                                            (A[u][c]==color)&&
                                            (A[u][d]==color)&&(u!=d))
                                                cand++;
                                    num += (cand*(cand-1))/2;
                                }
        return num;
    }

    // Book Bn is K2 joined with the empty graph on n vertices (all edges in between), hence Bn has n+2 vertices
    public long numB2(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                                cand++;
                    num += (cand*(cand-1))/2;
                }
        return num;
    }
    
    public long numB3(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2))/6;
                }
        return num;
    }
    
    public long numB4(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2)*(cand-3))/24;
                }
        return num;
    }
    
    public long numB5(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2)*(cand-3)*(cand-4))/120;
                }
        return num;
    }
    
    public long numB8(int color) {
        long num = 0;
        for (int a=0; a<n; a++)
            for (int b=a+1; b<n; b++)
                if (A[a][b]==color) {
                    int cand = 0;
                    for (int u=0; u<n; u++)
                        if ((u!=a)&&(A[u][a]==color)&&
                            (u!=b)&&(A[u][b]==color))
                                cand++;
                    num += (cand*(cand-1)*(cand-2)*(cand-3)*(cand-4)*(cand-5)*(cand-6)*(cand-7))/40320;
                }
        return num;
    }
    
    public long numC3(int color) {
        return numK3(color);
    }
    
    public long numC4(int color) {
        long num = 0;                       // a - b - d - c - a
        for (int a=0; a<n; a++)             // a is smallest
            for (int b=a+1; b<n; b++)       // b is smaller than c
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if ((A[a][c]==color)&&
                            (A[b][c]==color))
                            for (int d=a+1; d<n; d++)
                                if ((A[b][d]==color)&&(b!=d)&&
                                    (A[c][d]==color)&&(c!=d))
                                        num++;
        return num;
    }
    
    public long numC5(int color) {
        long num = 0;                       // a - b - d - e - c - a
        for (int a=0; a<n; a++)             // a is smallest
            for (int b=a+1; b<n; b++)       // b is smaller than c
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if (A[a][c]==color)
                            for (int d=a+1; d<n; d++)
                                if ((b!=d)&&(A[b][d]==color))
                                    for (int e=a+1; e<n; e++)
                                        if ((d!=e)&&(A[d][e]==color)&&
                                            (c!=e)&&(A[c][e]==color))
                                                num++;
        return num;
    }   

    public long numC6(int color) {
        long num = 0;                       // a - b - d - e - f - c - a
        for (int a=0; a<n; a++)             // a is smallest
            for (int b=a+1; b<n; b++)       // b is smaller than c
                if (A[a][b]==color)
                    for (int c=b+1; c<n; c++)
                        if (A[a][c]==color)
                            for (int d=a+1; d<n; d++)
                                if ((b!=d)&&(A[b][d]==color))
                                    for (int f=a+1; f<n; f++)
                                        if ((f!=c)&&(A[f][c]==color))
                                            for (int e=a+1; e<n; e++)
                                                if ((e!=d)&&(A[e][d]==color)&&
                                                    (e!=f)&&(A[e][f]==color))
                                                        num++;
        return num;                    
    }
    
    // Wheel Wn has n vertices: one vertex adjacent to all vertices of the cycle Cn-1
    public long numW4(int color) {
        return numK4(color);        
    }

    public long numW5(int color) {
        long num = 0;                               // a is adjacent to all
        for (int a=0; a<n; a++)                     //               a   a   a   a   a
            for (int b=0; b<n; b++)                 // the cycle C4: b - c - e - d - b
                if ((b!=a)&&(A[a][b]==color))       // b is smallest on C4
                    for (int c=b+1; c<n; c++)       // c is smaller than d
                        if ((A[a][c]==color)&&(c!=a)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&(d!=a)&&
                                    (A[b][d]==color))
                                    for (int e=b+1; e<n; e++)
                                        if ((A[a][e]==color)&&(e!=a)&&
                                            (A[c][e]==color)&&(e!=c)&&
                                            (A[d][e]==color)&&(e!=d))
                                                num++;
        return num;                            
    }

    public long numW6(int color) {
        long num = 0;                               // a is adjacent to all
        for (int a=0; a<n; a++)                     //               a   a   a   a   a   a
            for (int b=0; b<n; b++)                 // the cycle C5: b - c - e - f - d - b
                if ((b!=a)&&(A[a][b]==color))       // b is smallest on C5
                    for (int c=b+1; c<n; c++)       // c is smaller than d
                        if ((A[a][c]==color)&&(c!=a)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&(d!=a)&&
                                    (A[b][d]==color))
                                    for (int e=b+1; e<n; e++)
                                        if ((A[a][e]==color)&&(e!=a)&&
                                            (A[c][e]==color)&&(e!=c))
                                            for (int f=b+1; f<n; f++)
                                                if ((A[a][f]==color)&&(f!=a)&&
                                                    (A[d][f]==color)&&(f!=d)&&
                                                    (A[e][f]==color)&&(f!=e))
                                                        num++;
        return num;                            
    }

    public long numW7(int color) {
        long num = 0;                               // a is adjacent to all
        for (int a=0; a<n; a++)                     //               a   a   a   a   a   a   a
            for (int b=0; b<n; b++)                 // the cycle C6: b - c - e - g - f - d - b
                if ((b!=a)&&(A[a][b]==color))       // b is smallest on C6
                    for (int c=b+1; c<n; c++)       // c is smaller than d
                        if ((A[a][c]==color)&&(c!=a)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&(d!=a)&&
                                    (A[b][d]==color))
                                    for (int e=b+1; e<n; e++)
                                        if ((A[a][e]==color)&&(e!=a)&&
                                            (A[c][e]==color)&&(e!=c))
                                            for (int f=b+1; f<n; f++)
                                                if ((A[a][f]==color)&&(f!=a)&&
                                                    (A[d][f]==color)&&(f!=d))
                                                    for (int g=b+1; g<n; g++)
                                                        if ((A[a][g]==color)&&(g!=a)&&
                                                            (A[e][g]==color)&&(g!=e)&&
                                                            (A[f][g]==color)&&(g!=f))
                                                                num++;
        return num;                            
    }
    
    public long numW8(int color) {
        long num = 0;                               // a is adjacent to all
        for (int a=0; a<n; a++)                     //               a   a   a   a   a   a   a   a
            for (int b=0; b<n; b++)                 // the cycle C7: b - c - e - g - h - f - d - b
                if ((b!=a)&&(A[a][b]==color))       // b is smallest on C7
                    for (int c=b+1; c<n; c++)       // c is smaller than d
                        if ((A[a][c]==color)&&(c!=a)&&
                            (A[b][c]==color))
                            for (int d=c+1; d<n; d++)
                                if ((A[a][d]==color)&&(d!=a)&&
                                    (A[b][d]==color))
                                    for (int e=b+1; e<n; e++)
                                        if ((A[a][e]==color)&&(e!=a)&&
                                            (A[c][e]==color)&&(e!=c))
                                            for (int f=b+1; f<n; f++)
                                                if ((A[a][f]==color)&&(f!=a)&&
                                                    (A[d][f]==color)&&(f!=d))
                                                    for (int g=b+1; g<n; g++)
                                                        if ((A[a][g]==color)&&(g!=a)&&
                                                            (A[e][g]==color)&&(g!=e))
                                                            for (int h=b+1; h<n; h++)
                                                                if ((A[a][h]==color)&&(h!=a)&&
                                                                    (A[f][h]==color)&&(h!=f)&&
                                                                    (A[g][h]==color)&&(h!=g))
                                                                        num++;
        return num;                            
    }
}
