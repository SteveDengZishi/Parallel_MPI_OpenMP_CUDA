#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/
int i;
int j;
int nit = 0; /* number of iterations */
FILE * fp;
char output[100] ="";
int comm_sz;
int my_rank;
float error; /* The absolute relative error */
int num = 0;  /* number of unknowns */

/***** Globals ******/
float** a; /* The coefficients */
float* x;  /* The unknowns */
float* b;  /* The constants */

float* _error; // depending on how many lines is it taking in
float* _x;  //local x, depending on how many lines
float* global_error;
int calculated_error = 0; // The calculated error

int lines;
int send_count;

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

//checking all lines in current process for whether error falls into acceptable range
int check_error(){
    int stop_proc = 1;
    for(i=0;i<num;i++){
        if (global_error[i]>error) stop_proc = 0;
    }
    return stop_proc;
}

//debug functions
void print_matrix_a(){
    printf("printing a from process %d\n", my_rank);
    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            printf("%f ",a[i][j]);
        }
    }
    printf("\n");
}

void print_matrix_x(){
    printf("printing x from process %d\n", my_rank);
    for(i=0;i<num;i++) printf("%f ",x[i]);
    printf("\n");
}

void print_matrix_b(){
    printf("printing b from process %d\n", my_rank);
    for(i=0;i<num;i++) printf("%f ",b[i]);
    printf("\n");
}


/************************************************************/


int main(int argc, char *argv[])
{

    //mpi program init
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(argc != 2){
        printf("Usage: ./gsref filename\n");
        exit(1);
    }

    //Read for all processes to prevent communication
    get_input(argv[1]);

    //printf("it is good so far from process %d\n", my_rank);
    /* Check for convergence condition */
    /* This function will exit the program if the coffeicient will never converge to
     * the needed absolute error.
     * This is not expected to happen for this programming assignment.
     */
    check_matrix();
    //print_matrix_a();
    //print_matrix_x();
    //print_matrix_b();

    //calculating task for each processes
    //only after get_input so num & error is defined
    lines = num/comm_sz;    
    send_count = lines*num;
    //printf("The line count in process %d is %d\n", my_rank,lines);

    //check whether the task is divisible, quit if not the case
    if(num % comm_sz != 0 && my_rank==0){
        printf("\nNumber of unknown should be perfectly divisible by number of processes!\n");
        printf("Please check your input before you proceed\n");
    }

    if(num % comm_sz != 0){
        exit(1);
    }

    //allocate memory for all local vars according to lines
    _error = malloc(lines*sizeof(float));
    for(i=0;i<lines;i++) _error[i]=1.0;
    _x = malloc(lines*sizeof(float));

    //in different processes, use vars according to bounds
    while(calculated_error == 0){
        //if there is one error does not satisfy the requirement, go to the next iteration
        nit++;
        int offset = my_rank * lines;
        //printf("The offset in process %d is %d\n", my_rank,offset);

        for(i=0;i<lines;i++){

            float sum = 0;

            for(j=0;j<num;j++){
                if((i+offset)!=j) sum += a[i+offset][j] * x[j];
            }

            //printf("The sum in process %d is %f\n", my_rank,sum);
            _x[i] = (b[i+offset] - sum) / a[i+offset][i+offset];

            //int k;
            //for(k=0;k<lines;k++) printf("The _x in process %d is %f\n", my_rank,_x[k]);
            _error[i] = fabs((_x[i]-x[i+offset]) / _x[i]);
            //for(k=0;k<lines;k++) printf("The _e in process %d is %f\n", my_rank,_error[k]);
        }

        //need to gather all global error and check continuation condition
        global_error = malloc(num*sizeof(float));

        MPI_Allgather(_error,lines,MPI_FLOAT,global_error,lines,MPI_FLOAT,MPI_COMM_WORLD);
        //check error to update and determine whether to continue 
        //it must be done in the global scope
        calculated_error = check_error();
        //gather all values from x to update in the next iteration
        MPI_Allgather(_x,lines,MPI_FLOAT,x,lines,MPI_FLOAT,MPI_COMM_WORLD);
        //printf("In process %d the x after all gather is\n", my_rank);
        //print_matrix_x();
        free(global_error);
    }

    //after the loop finished, all processes reaches criteria for all lines
    //call reduce to get the maximum number of iterations
    //int max;
    //MPI_Reduce(&nit,&max,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);

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

    }

    MPI_Finalize();

    return 0;
}
