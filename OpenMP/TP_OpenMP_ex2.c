#include <stdio.h>
#include <omp.h>

int main(){
	int valeur1 = 1000;
	int valeur2 = 2000;
	#pragma omp parallel num_threads(4) firstprivate(valeur2)
	{
		valeur2++;
		printf("I'm thread n° %d and got values %d %d\n",omp_get_thread_num ( ) ,valeur1,valeur2);
	}
	return 0;
}