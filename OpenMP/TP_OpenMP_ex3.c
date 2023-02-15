#include <stdio.h>
#include <omp.h>

int main(){
	int i;
	#pragma omp parallel for
	for(i = 0 ; i < 50 ; i++)
	{
		printf("I'm thread n° %d and got index %d \n",omp_get_thread_num ( ) ,i);
	}
	return 0;
}