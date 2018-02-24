#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float** a; /* The coefficients */
float* x;  /* The unknowns */
float* b;  /* The constants */
float error; /* The absolute relative error */
int num = 0;  /* number of unknowns */



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/*
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
    int bigger = 0; /* Set to 1 if at least one diag element > sum  */
    int i, j;
    float sum = 0;
    float aii = 0;

    for(i = 0; i < num; i++)
    {
        sum = 0;
        aii = fabs(a[i][i]);

        for(j = 0; j < num; j++)
            if( j != i)
                sum += fabs(a[i][j]);

        if( aii < sum)
        {
            printf("The matrix will not converge.\n");
            exit(1);
        }

        if(aii > sum)
            bigger++;

    }

    if( !bigger )
    {
        printf("The matrix will not converge\n");
        exit(1);
    }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * error will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
    FILE * fp;
    int i,j;

    fp = fopen(filename, "r");
    if(!fp)
    {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    fscanf(fp,"%d ",&num);
    fscanf(fp,"%f ",&error);

    /* Now, time to allocate the matrices and vectors */
    a = (float**)malloc(num * sizeof(float*));
    if( !a)
    {
        printf("Cannot allocate a!\n");
        exit(1);
    }

    for(i = 0; i < num; i++)
    {
        a[i] = (float *)malloc(num * sizeof(float));
        if( !a[i])
        {
            printf("Cannot allocate a[%d]!\n",i);
            exit(1);
        }
    }

    x = (float *) malloc(num * sizeof(float));
    if( !x)
    {
        printf("Cannot allocate x!\n");
        exit(1);
    }


    b = (float *) malloc(num * sizeof(float));
    if( !b)
    {
        printf("Cannot allocate b!\n");
        exit(1);
    }

    /* Now .. Filling the blanks */

    /* The initial values of Xs */
    for(i = 0; i < num; i++)
        fscanf(fp,"%f ", &x[i]);

    for(i = 0; i < num; i++)
    {
        for(j = 0; j < num; j++)
            fscanf(fp,"%f ",&a[i][j]);

        /* reading the b element */
        fscanf(fp,"%f ",&b[i]);
    }

    fclose(fp);

}


/************************************************************/


int main(int argc, char *argv[])
{

    int i;
    int nit = 0; /* number of iterations */
    FILE * fp;
    char output[100] ="";
    int comm_sz;
    int my_rank;
    double** _a;
    double* _x;
    double* _b;

    //mpi program init
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_size(MPI_COMM_WORLD, &my_rank);

    //calculating task for each processes
    int lines = num/comm_sz;
    int send_count = lines*num;

    //allocating local variables for each processes
    _a = malloc(lines*sizeof(double*));
    for(i=0;i<lines;i++){
        _a[i]=malloc(num*sizeof(double));
    }

    _x = malloc(num*sizeof(double));
    _b = malloc(lines*sizeof(double));


    //master program taking input, check convergence and scatter vectors to all processes
    if (my_rank==0)
    {
        if( argc != 2)
        {
            printf("Usage: ./gsref filename\n");
            exit(1);
        }

        /* Read the input file and fill the global data structure above */
        get_input(argv[1]);

        /* Check for convergence condition */
        /* This function will exit the program if the coffeicient will never converge to
         * the needed absolute error.
         * This is not expected to happen for this programming assignment.
         */
        check_matrix();

        //scatter input to all processes
        MPI_Scatter(a,send_count,MPI_DOUBLE,_a,send_count,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatter(x,num,MPI_DOUBLE,_x,num,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatter(b,lines,MPI_DOUBLE,_b,lines,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    else {
        MPI_Scatter(a,send_count,MPI_DOUBLE,_a,send_count,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatter(x,num,MPI_DOUBLE,_x,num,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Scatter(b,lines,MPI_DOUBLE,_b,lines,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    //do calculation according to the formula

    //gather the data from processes to master

    //do condition checking for each processes

    //master process write to files
    if(my_rank==0)
    {
        /* Writing results to file */
        sprintf(output,"%d.sol",num);
        fp = fopen(output,"w");
        if(!fp)
        {
            printf("Cannot create the file %s\n", output);
            exit(1);
        }

        for( i = 0; i < num; i++)
            fprintf(fp,"%f\n",x[i]);

        printf("total number of iterations: %d\n", nit);

        fclose(fp);

        exit(0);
    }
    MPI_Finalize();
}