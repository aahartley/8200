/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
*	Assumptions:    A, B, and C are square matrices n by n; 
*			the total number of processors (np) is a square number (q^2).
*	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*********************************************************************************************/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include "mpi.h"
//#include <string.h>
#define min(a, b) ((a < b) ? a : b)
#define SZ 2000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.


/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
        array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
        for (i=1; i<n_rows; i++){
                array[i] = array[0] + i * n_cols;
        }
        return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}
void testInit(double **A, double **B,double **C, int blck_sz, double **testA, double **testB, int rank){

	for (int i=0; i<SZ; i++){
                for (int j=0; j<SZ; j++){
                        if(i==j){
                                testA[i][j]= 1.0;
                                if(i+1 < SZ)
                                        testA[i+1][j]=1.0;
                                testB[i][j]=1.0;
                                if(j+1 < SZ)
                                        testB[i][j+1]=1.0;
                                
                        }
                        else{
                                if(testA[i][j]!=1)testA[i][j]=0.0;
                                if(testB[i][j]!=1)testB[i][j]=0.0;
                                
                        }
                }
        }

	for (int i=0; i<blck_sz; i++){
		for (int j=0; j<blck_sz; j++){
			if(rank==0 || rank ==1 || rank ==2){
                        	A[i][j]= testA[i][j+rank*blck_sz];
                        	B[i][j] = testB[i][j+rank*blck_sz];
                        	C[i][j]=0.0;
                        }
                        if(rank==3 || rank ==4 || rank ==5){
                        	A[i][j]= testA[i+blck_sz][j+(rank-3)*blck_sz];
                        	B[i][j] = testB[i+blck_sz][j+(rank-3)*blck_sz];
                        	C[i][j]=0.0;
                        }
                        if(rank==6 || rank ==7 || rank ==8){
                        	A[i][j]= testA[i+blck_sz*2][j+(rank-3*2)*blck_sz];
                        	B[i][j] = testB[i+blck_sz*2][j+(rank-3*2)*blck_sz];
                        	C[i][j]=0.0;
                        }
		}
	}
}

void copyMat(double **buff, double **orig, int block_sz){
	for(int i=0; i<block_sz;i++){
		for(int j=0; j<block_sz;j++){
			buff[i][j] = orig[i][j];
		}
	}
}
void matmulAdd(double **c, double **a, double **b, int block_sz){
    	for(int i=0; i<block_sz; i++){
		for(int j=0; j<block_sz; j++){
			for(int k=0; k<block_sz; k++){
				
				c[i][j] = c[i][j] + (a[i][k] * b[k][j]);
			}
		}
	}
}
/**
*	Perform the SUMMA matrix multiplication. 
*       Follow the pseudo code in lecture slides.
*/
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
						double **my_B, double **my_C){
        double **buffA, **buffB;
	buffA = alloc_2d_double(block_sz, block_sz);
	buffB = alloc_2d_double(block_sz, block_sz);
        MPI_Comm grid_comm;
	MPI_Comm row_comm;
	MPI_Comm col_comm;
	int dimsizes[2];
	int wraparound[2];
        int coordinates[2];
        int reorder = 1;
	int free_coords[2];
	int my_grid_rank, grid_rank;
	dimsizes[0] = dimsizes[1] = proc_grid_sz;
	wraparound[0] = wraparound[1] = 1;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_grid_rank);
	MPI_Cart_coords(grid_comm, my_grid_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);
        //create row
	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm,free_coords, &row_comm);
	//create col
	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);
	//Add your implementation of SUMMA algorithm
	for(int k=0; k < proc_grid_sz; k++)
	{
	    if(coordinates[1]==k){
		copyMat(buffA, my_A, block_sz);
		//buffA = my_A;
	    }
	    MPI_Bcast(&buffA[0][0],block_sz*block_sz, MPI_DOUBLE, k, row_comm);
	    if(coordinates[0]== k){
		//buffB = my_B;
		copyMat(buffB, my_B, block_sz);
	    }
	    MPI_Bcast(&buffB[0][0], block_sz*block_sz, MPI_DOUBLE, k , col_comm);
	    //printf("%d %d %d %d \n",my_grid_rank, my_rank, coordinates[0], coordinates[1]);
	    if(coordinates[0] == k && coordinates[1] ==k){
                matmulAdd(my_C, my_A, my_B, block_sz);
	    }
	    else if(coordinates[0]==k){
                matmulAdd(my_C, buffA, my_B, block_sz);
	    }
	    else if(coordinates[1]==k){
                matmulAdd(my_C, my_A, buffB, block_sz);
	    }
	    else{
	        matmulAdd(my_C, buffA, buffB, block_sz);
	    }
	}


}


int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides


	
	srand(time(NULL));							// Seed random numbers
        MPI_Status status;
 	int tag;
/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);


/* assign values to 1) proc_grid_sz and 2) block_sz*/
	
        proc_grid_sz = (int)sqrt((double)num_proc);
	block_sz = SZ/proc_grid_sz;
	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process
	double **testA, **testB;	
	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);
	testA = alloc_2d_double(SZ, SZ);
	testB = alloc_2d_double(SZ,SZ);
	
	initialize(A, B, C, block_sz);
	//testInit(A,B,C,block_sz,testA,testB,rank);
	
	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();


	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);


	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;
	
	// Insert statements for testing
	/*
	printf("\n");
	//print A
	if (rank == 0){
		for (int i = 0; i < block_sz; i++){
			for (int j = 0; j < block_sz; j++)
				printf("%f ", A[i][j]);
			printf("\n");
		}
	}
	//print B
	printf("\n");
	if (rank == 0 ){
                for (int i = 0; i < block_sz; i++){
                        for (int j = 0; j < block_sz; j++)
                                printf("%f ", B[i][j]);
                        printf("\n");
                }
        }
	
	printf("\n");
	//check diagonal 0
	if (rank == 0 ){
                for (int i = 0; i < block_sz; i++){
                        for (int j = 0; j < block_sz; j++)
                                printf("%f ", C[i][j]);
                        printf("\n");
                }
        }
	printf("\n");
	//check diagonal 4
        if (rank == 4 ){
                for (int i = 0; i < block_sz; i++){
                        for (int j = 0; j < block_sz; j++)
                                printf("%f ", C[i][j]);
                        printf("\n");
                }
        }
	printf("\n");
	//check diagonal 8
        if (rank == 8 ){
                for (int i = 0; i < block_sz; i++){
                        for (int j = 0; j < block_sz; j++)
                                printf("%f ", C[i][j]);
                        printf("\n");
                }
        }
	*/
	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
			SZ, num_proc, total_time);
	}

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}
