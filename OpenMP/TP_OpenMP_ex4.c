#include<stdio.h>
#include<omp.h>
#define CHUNK 1000 //no advantage for CHUNK = 100, dynamic at clear disadvantage for CHUNK = 10 & =1, static performs as without schedule

double monoProcRef(const long nb_pas){
	double pas,t1,t2;
	int i; double x, pi, som = 0.0;
	pas = 1.0/(double) nb_pas;
	t1=omp_get_wtime();
	for (i=0;i< nb_pas; i++){
		x = (i + 0.5)*pas;
	som = som + 4.0/(1.0+x*x);
	}
	pi = pas * som;
	t2=omp_get_wtime();
	printf("PI=%f %f \n",pi,t2-t1);
	return t2-t1;
}

double parallelProc(const long nb_pas, unsigned int nbThreads){//errors in pi and elapsed time because of shared double x
	double pas,t1,t2;
	int i; double pi, som = 0.0;
	pas = 1.0/(double) nb_pas;
	t1=omp_get_wtime();
	#pragma omp parallel for reduction(+:som) num_threads(nbThreads)
	for (i=0;i< nb_pas; i++)
	{
		som +=  4.0/(1.0+(i + 0.5)*pas*(i + 0.5)*pas);
	}
	pi = pas * som;
	t2=omp_get_wtime();
	printf("PI=%f %f \n",pi,t2-t1);
	return t2-t1;
}

double parallelProcStatic(const long nb_pas, unsigned int nbThreads){//errors in pi and elapsed time because of shared double x
	double pas,t1,t2;
	int i; double pi, som = 0.0;
	pas = 1.0/(double) nb_pas;
	t1=omp_get_wtime();
	#pragma omp parallel for reduction(+:som) schedule(static,CHUNK) num_threads(nbThreads)
	for (i=0;i< nb_pas; i++)
	{
		som +=  4.0/(1.0+(i + 0.5)*pas*(i + 0.5)*pas);
	}
	pi = pas * som;
	t2=omp_get_wtime();
	printf("PI=%f %f \n",pi,t2-t1);
	return t2-t1;
}

double parallelProcDynamic(const long nb_pas, unsigned int nbThreads){//errors in pi and elapsed time because of shared double x
	double pas,t1,t2;
	int i; double pi, som = 0.0;
	pas = 1.0/(double) nb_pas;
	t1=omp_get_wtime();
	#pragma omp parallel for reduction(+:som) schedule(dynamic,CHUNK) num_threads(nbThreads)
	for (i=0;i< nb_pas; i++)
	{
		som +=  4.0/(1.0+(i + 0.5)*pas*(i + 0.5)*pas);
	}
	pi = pas * som;
	t2=omp_get_wtime();
	printf("PI=%f %f \n",pi,t2-t1);
	return t2-t1;
}


int main () {
	static long nb_pas = 1000000000;
	unsigned int nbIter = 1;
	double monoProcTime = 0.;
	for (unsigned int i = 0; i < nbIter; i++){
		monoProcTime+=monoProcRef(nb_pas);
	}
	unsigned int threads[] = {1,2,3,4,6,8,10,12,16,20,24,32,48,64,96,128,256,0};
	double parallelTime[sizeof(threads)/sizeof(unsigned int)]={0.},
			parallelTimeStatic[sizeof(threads)/sizeof(unsigned int)]={0.},
			parallelTimeDynamic[sizeof(threads)/sizeof(unsigned int)]={0.};
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		for (unsigned int i = 0; i < nbIter; i++){
			parallelTime[j]+=parallelProc(nb_pas,threads[j]);
		}
		parallelTime[j]/=nbIter;
	}
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		for (unsigned int i = 0; i < nbIter; i++){
			parallelTimeStatic[j]+=parallelProcStatic(nb_pas,threads[j]);
		}
		parallelTimeStatic[j]/=nbIter;
	}
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		for (unsigned int i = 0; i < nbIter; i++){
			parallelTimeDynamic[j]+=parallelProcDynamic(nb_pas,threads[j]);
		}
		parallelTimeDynamic[j]/=nbIter;
	}
	
	monoProcTime/=nbIter;
	printf(" mono processor mean time : %f\n",monoProcTime);
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		printf(" nb threads : %d parallel mean time : %f\n",threads[j],parallelTime[j]);
	}
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		printf(" nb threads : %d parallel static mean time : %f\n",threads[j],parallelTimeStatic[j]);
	}
	for (unsigned int j = 0 ; j < sizeof(threads)/sizeof(unsigned int) ; j++){
		printf(" nb threads : %d parallel dynamic mean time : %f\n",threads[j],parallelTimeDynamic[j]);
	}
		 
}