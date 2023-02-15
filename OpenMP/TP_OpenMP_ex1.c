#include <stdio.h>
#include <omp.h>

int main(){
	//in bash : export OMP_NUM_THREADS 4
	/*omp_set_num_threads(4);
	#pragma omp parallel*/
	#pragma omp parallel num_threads(4)
	{
		printf("I'm thread n° %d\n",omp_get_thread_num ( ) );
	}
	return 0;
}