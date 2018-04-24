#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]){

	//global variables
	FILE* fp;
	char filename[100]="";
	int* num;
	int* crossed;
	int i;
	int j;

	//get N and number of threads
	int upper_bound = strtol(argv[1], NULL, 10);
	int thread_count = strtol(argv[2], NULL, 10);

	//init time
	double tstart = 0.0, tend = 0.0, ttaken = 0.0;

	//start timing
	tstart = omp_get_wtime();

	//generating prime numbers
	int init_count = upper_bound-1;
	num = malloc((init_count)*sizeof(int));
	crossed = malloc((init_count)*sizeof(int));

	#pragma omp parallel for num_threads(thread_count) private(i)
	for(i = 0; i < init_count; i++){
		num[i] = i+2;
		crossed[i] = 0;
	}

	int check_bound = (upper_bound+1)/2;

	#pragma omp parallel for num_threads(thread_count) private(i,j)
	for(i = 2; i < check_bound; i++){
		if(crossed[i-2]==0){
			int multiple = upper_bound/i;
			for(j = 2; j <= multiple; j++){
				int value = j*i;
				crossed[value-2]=1;
			}
		}
	}

	//end timing
	tend = omp_get_wtime();
	ttaken = tend-tstart;
	printf("Time taken for the main part: %f\n", ttaken);

	// Writing results to file 
    sprintf(filename,"%d.txt",upper_bound);
    fp = fopen(filename,"w");
    if(!fp)
    {
        printf("Cannot create the file %s\n", filename);
        exit(1);
    }

    //printing prime numbers one at a line
    int prev=2;
    int counter=0;
    for(i = 0; i < init_count; i++){
    	if(crossed[i]==0) {
    		counter++;
    		int diff = num[i]-prev;
    		prev = num[i];
    		fprintf(fp,"%d, %d, %d\n", counter, num[i], diff);
    	}
    }

    //close file
    fclose(fp);
	
}
