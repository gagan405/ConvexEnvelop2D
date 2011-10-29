/*********************************************************************************************
		Convex Envelope computation in a 2 dimensional plane
		Version 1.0
Gagan Bihari MIshra					Chiranjeeb Basak

Advisor : Dr. Mireille Gregoire
Institute of Parallel and Distributed Systems
University of Stuttgart, Germany	

**********************************************************************************************/

//NPOINTS points in 2D with known coordinates x,y
//hypothesis: no 3 points are aligned
// we want to calculate the convex envelope of those points
//N.B.: there are are algorithms that are much more efficient than this one,
//but that are also more complicated to parallelize

#include "cutil.h"
#include <stdio.h>

#define NPOINTS 2048 		//number of points
#define TYPE int

#define blockSizeX 256
#define blockSizeY 2


TYPE h_x[NPOINTS];		//x coordinate of points
TYPE h_y[NPOINTS];		//y coordinate of points
int h_edges[NPOINTS*2];		//valid edges
int h_edges_tmp[NPOINTS*2];	//valid edges
int h_res_device[NPOINTS+1];
int h_res[NPOINTS+1];		//result: indexes in order of the different points of the convex envelope
				//can be as long as NPOINTS (the points form a convex polygon)

//there are either 0 or 2 valid edges starting from a point i
//if 0 -> both values at position 2*i and 2*i+1 are -1
//if 2-> the indexes of the other ends from the 2 edges

/****************************


GPU code begins here --


*****************************/

__global__ void find_edges_on_device(TYPE * h_x, TYPE * h_y, int *h_edges){
	
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<NPOINTS; i+=blockDim.x*gridDim.x){
		for (int j=threadIdx.y+blockIdx.y*blockDim.y; j<NPOINTS; j+=blockDim.y*gridDim.y)
		{
			if (i>=j)
			{
			continue;  //edge i,j == edge j,i
			}
		
			//all the others points should be on the same side of the edge i,j
			
			//normal to the edge (unnormalized)

			TYPE nx= - ( h_y[j]- h_y[i]);
			TYPE ny= h_x[j]- h_x[i];
		
			int k=0;
			while ((k==i)||(k==j))
			{
				k++;
			} //k will be 0,1,or 2, but different from i and j to avoid 
			
			long scalarProduct=nx* (h_x[k]-h_x[i])+ny* (h_y[k]-h_y[i]);
		
			if (scalarProduct<0)
			{
				nx*=-1;
				ny*=-1;
			}

			//we have now at least one point with scalarProduct>0
			//all the other points should comply with the same condition for
			//the edge to be valid
		
			bool isValid=true;

			//loop on all the points 

			for (int k=0; k<NPOINTS; k++)
			{
				scalarProduct=nx* (h_x[k]-h_x[i])+ny* (h_y[k]-h_y[i]);
				if (scalarProduct <0)
				{	//invalid edge
					isValid = false;
					break;
				}
			}
			
			if (isValid)
			{
				int tmp_i = i;
				int tmp_j = j;

			//we use atomic functions to write to the global memory
			//as two threads might be writing to the same location

				if( -1 != atomicCAS(&h_edges[2*i], -1, tmp_j) )
					h_edges[2*i+1]=j;

				if( -1 != atomicCAS(&h_edges[2*j], -1, tmp_i) )
					h_edges[2*j+1]=i;
			}
		}
	}
}


/****************************


Host code begins here --


*****************************/


void find_edges_on_host(TYPE * h_x, TYPE * h_y, int *h_edges)
{
	//loop on all possible edges == pairs of points
	for (int i=0; i<NPOINTS; i++)
	{
		for (int j=0; j<NPOINTS; j++)
		{
			if (i>=j)
			{
			continue;  //edge i,j == edge j,i
			}
		
			//all the others points should be on the same side of the edge i,j
			
			//normal to the edge (unnormalized)
			TYPE nx= - ( h_y[j]- h_y[i]);
			TYPE ny= h_x[j]- h_x[i];
		
			int k=0;
			while ((k==i)||(k==j))
			{
				k++;
			} //k will be 0,1,or 2, but different from i and j to avoid 
			//scalarProduct=0
		
			TYPE scalarProduct=nx* (h_x[k]-h_x[i])+ny* (h_y[k]-h_y[i]);
			if (scalarProduct<0)
			{
				nx*=-1;
				ny*=-1;
			}
			//we have now at least one point with scalarProduct>0
			//all the other points should comply with the same condition for
			//the edge to be valid
		
			bool isValid=true;
			//loop on all the points 
			for (int k=0; k<NPOINTS; k++)
			{
				scalarProduct=nx* (h_x[k]-h_x[i])+ny* (h_y[k]-h_y[i]);
				if (scalarProduct <0)
				{	//invalid edge
					isValid = false;
					break;
				}
			}
			
			if (isValid)
			{
				//write the edge to h_edges in the first available position
				if (h_edges[2*i]==-1)
				{
					h_edges[2*i]=j;// atomic
					//do a check if the value is updated

				}
				else
				{
					h_edges[2*i+1]=j;
				}
				
				//we write the edge two times for a direct access in the next stage
				if (h_edges[2*j]==-1)
				{
					h_edges[2*j]=i;
				}
				else
				{
					h_edges[2*j+1]=i;
				}
				
			}
		}
	}
	return;
}

/****************************

GPU calculation can result in an array which is inverted version of the host result.
In such a case, we have to revert the array before comparing with the CPU result.

*****************************/

void reverse_array(int *h_res_device, int *reversed_dev_result)
{	
	int i = NPOINTS;
	int j = 0;
	
	while(h_res_device[i] == -1){
		reversed_dev_result[i] = -1;
		i--;
	}
	while(i >= 0){
		reversed_dev_result[j] = h_res_device[i];
		j++;
		i--;
	}

}
/****************************

This function sorts the array returned by the host/GPU function and finds out the 
points which form the convex polygon

*****************************/
int sort_edges(int * h_edges, int * h_res)
{
	// find the first point that belongs to the convex envelope
	int i0=0;
	int lastValue = 0;
	while ((i0<NPOINTS)&&(h_edges[2*i0]==-1))
	{
		i0++;
	}
	
	//i0 now belongs to the envelope
	
	h_res[0]=i0;
	h_res[1]=h_edges[2*i0];
	int k=2; //index in the envelope
	lastValue=h_res[1];
	while (k<=NPOINTS)
	{
		//follow the edges, take the points that are not already in h_res
		if (h_edges[2*lastValue]==h_res[k-2])
		{
			h_res[k]=h_edges[2*lastValue+1];
		}
		else
		{
			h_res[k]=h_edges[2*lastValue];
		}
		lastValue=h_res[k];
		if (h_res[k]==i0)
		{
			break;			
		}
		k++;
	}
	
	return k;

}

/****************************

This function compares two arrays passed as arguements

*****************************/
bool check(int *a,int* b){
  for(int i=0;i<NPOINTS+1;i++){
        if(a[i]!=b[i]){
    		return true;
   	}
  }
  return false;
}

/****************************


The main function


*****************************/


int main(int argc, char** args  )
{

  TYPE *dX;
  TYPE *dY;
  int *d_edges;
  int *reversed_dev_result;

//we need timers to measure the execution times

  unsigned int timer1=0;
  unsigned int timer2=0;

//allocate memory in the device

  cudaMalloc((void **)&dX,NPOINTS*sizeof(TYPE));
  cudaMalloc((void **)&dY,NPOINTS*sizeof(TYPE));
  cudaMalloc((void **)&d_edges,NPOINTS*2*sizeof(int));

  CUT_SAFE_CALL(cutCreateTimer(&timer1));
  CUT_SAFE_CALL(cutCreateTimer(&timer2));

//initialisation of the coordinates

for (int i=0; i<NPOINTS; i++)//loop on points
{
	h_x[i]=(rand()%100000);//10000.0f-0.5f;
	h_y[i]=(rand()%100000);//10000.0f-0.5f;
	h_edges[i*2]=-1; 		//no valid edge by default
	h_edges[i*2+1]=-1;
	h_edges_tmp[i*2]=-1; 		//no valid edge by default
	h_edges_tmp[i*2+1]=-1; 
    	h_res[i+1]=-1;
	h_res_device[i+1]=-1;
	
}


//Copy the arrays from the host to the device

  cudaMemcpy(dX,h_x,NPOINTS*sizeof(TYPE),cudaMemcpyHostToDevice);  
  cudaMemcpy(dY,h_y,NPOINTS*sizeof(TYPE),cudaMemcpyHostToDevice);  
  cudaMemcpy(d_edges,h_edges_tmp,NPOINTS*2*sizeof(int),cudaMemcpyHostToDevice);  

//Set the grid and block dimensions before calling the kernel

  dim3 bS(blockSizeX,blockSizeY,1);		
  dim3 gS((NPOINTS+blockSizeX-1)/blockSizeX,(NPOINTS+blockSizeY-1)/blockSizeY,1);

//Start the timer
  CUT_SAFE_CALL(cutStartTimer(timer1));

//Call the CUDA Kernel
  find_edges_on_device<<<gS,bS>>>(dX,dY,d_edges);
  cudaThreadSynchronize();

//Stop the timer
  CUT_SAFE_CALL(cutStopTimer(timer1));

//Copy the result back to the Host
  cudaMemcpy(h_edges_tmp,d_edges,NPOINTS*2*sizeof(int),cudaMemcpyDeviceToHost);  

//find out which edges belong to the convex envelope in the host function
CUT_SAFE_CALL(cutStartTimer(timer2));

find_edges_on_host(h_x, h_y, h_edges);

CUT_SAFE_CALL(cutStopTimer(timer2));

float time1=cutGetAverageTimerValue(timer1);
float time2=cutGetAverageTimerValue(timer2);
printf(" Time on Device %f ms \t  Time on Host %f ms\n",time1, time2);

//reset timer1 and timer2
CUT_SAFE_CALL(cutResetTimer(timer1));
CUT_SAFE_CALL(cutResetTimer(timer2));

//sort edges found in host
int nedges=sort_edges(h_edges, h_res);

//sort edges found in device

nedges=sort_edges(h_edges_tmp, h_res_device);

//compare the host and device results

if(check(h_res,h_res_device)){
	
	//if they mismatch, possibly the device result is the reverse of host result
	//in such a case, reverse one of the array and compare again

	reversed_dev_result = (int *)malloc((NPOINTS+1) * sizeof(int));
	reverse_array(h_res_device, reversed_dev_result);
	
	//after reversing, compare again

	if(check(h_res,reversed_dev_result))
		printf("\nError!\n");
	else
		printf("\nPassed!\n");
	
	//Don't forget to free the allocated memory
	free(reversed_dev_result);
  }
  else
    printf("\nPassed!\n");

//free the allocated memory in device

cudaFree(dX);  
cudaFree(dY);  
cudaFree(d_edges);  

return 0;
}
