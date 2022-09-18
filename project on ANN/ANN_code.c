#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#define Max_E pow(10,-4)

int main()
{
	int i,j,k,p;    //for iterations
	int L,M,N,P;
	float e,Mean_sq_E;
	float I[100][100],IH[100][100],OH[100][100],IO[100][100],OO[100][100],TO[100][100];
	float V[100][100],W[100][100],del_W[100][100],del_V[100][100];
	float I_min[100],I_max[100],TO_min[100],TO_max[100];

	FILE *IP, *OP_1, *OP_2;

		IP  = fopen("IP.txt","r");
		OP_1  = fopen("OP.txt","w");
		OP_2 = fopen("Result.txt","w");

	
	fscanf(IP,"%d%d%d%d",&P,&L,&M,&N);
	fprintf(OP_2,"Total No of Patterns (P)=%d\n",P);
	fprintf(OP_2,"Total No of Input Neurons (L)=%d\n",L);	
	fprintf(OP_2,"Total No of Hidden Neurons (M)=%d\n",M);	
	fprintf(OP_2,"Total No of Output Neurons (N)=%d\n",N);
	
//Scanning & Printing Inputs for Input layer

	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			fscanf(IP,"%f",&I[i][p]);
		}
	}
	
	fprintf(OP_2,"\nI matrix of order %dX%d :\n",L,P);
	
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			fprintf(OP_2,"I[%d][%d]=%f\t",i,p,I[i][p]);
		}
		fprintf(OP_2,"\n");
	}
	
//Scanning & Printing Target output for Output layer

	fprintf(OP_2,"\nTO matrix of order %dX%d :\n",P,N);
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<=N;k++)
		{
			fscanf(IP,"%f",&TO[k][p]);
			fprintf(OP_2,"TO[%d][%d]:%f\t",k,p,TO[k][p]);
		}
		fprintf(OP_2,"\n");
	}

//Normalization of Inputs for Input layer

	for(i=1;i<=L;i++)
	{
		I_max[i]=-1000;I_min[i]=1000;
		
		for(p=1;p<=P;p++)
		{
			if(I[i][p]>I_max[i])
			I_max[i]=I[i][p];
			if(I[i][p]<I_min[i])
			I_min[i]=I[i][p];
		}
	}
	
	for(p=1;p<=P;p++)
	{
		for(i=1;i<=L;i++)
		{
			I[i][p]=0.1+0.8*((I[i][p]-I_min[i])/(I_max[i]-I_min[i]));
		}
	}
	
	fprintf(OP_2,"\nNormalized I matrix of order %dX%d :\n",L,P);
	
	for(i=1;i<=L;i++)
	{
		for(p=1;p<=P;p++)
		{
			fprintf(OP_2,"%f\t",I[i][p]);
		}
		fprintf(OP_2,"\n");
	}


//Normalization of Target output for Output layer

for(k=1;k<N+1;k++)
	{
		TO_max[k]=-1000;TO_min[k]=1000;
		for(p=1;p<=P;p++)
		{
			if(TO[k][p]>TO_max[k])
			TO_max[k]=TO[k][p];
			if(TO[k][p]<TO_min[k])
			TO_min[k]=TO[k][p];			
		}
	}
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<=N;k++)
		{
			TO[k][p]=-0.1+(1.0*((TO[k][p]-TO_min[k])/(TO_max[k]-TO_min[k])));
		}
	}

	fprintf(OP_2,"\nNormalized TO matrix of order %dX%d :\n",P,N);
	
	for(p=1;p<=P;p++)
	{
		for(k=1;k<N+1;k++)
		{
			fprintf(OP_2,"TO[%d][%d]:%f\t",k,p,TO[k][p]);
		}
		fprintf(OP_2,"\n");
	}

srand(time(NULL));
	
//Define V

	fprintf(OP_2,"\nV matrix of order %dX%d :\n",L+1,M);

	for(i=0;i<L+1;i++)
	{
		for(j=1;j<=M;j++)
		{
			if(i==0)
			{
				V[i][j]=0;
			}
			else
			{
				V[i][j]=1.0*rand()/RAND_MAX;
			}
		}
	}
	
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			fprintf(OP_2,"V[%d][%d]:%f\t",i,j,V[i][j]);
		}
		fprintf(OP_2,"\n");
	}
	fprintf(OP_2,"\n");

//Define W

	fprintf(OP_2,"\nW matrix of order %dX%d :\n",M+1,N);
	
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			if(j==0)
			{
				W[i][j]=0;
			}
			else
			{
			W[j][k]=1.0*rand()/RAND_MAX;
			}
		}
	}
	
	for(j=0;j<M+1;j++)
	{
		for(k=1;k<=N;k++)
		{
			fprintf(OP_2,"W[%d][%d]:%f\t",j,k,W[j][k]);
		}
		fprintf(OP_2,"\n");
	}

	int C=1;	//C=Counter

//Do-While loop TRAINING of Patterns

	do
	{		
		//Calculation for forward pass
		
		for(p=1;p<=P-5;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+(1.0);
				OH[j][p]=1/(1+exp(-IH[j][p]));
				IH[j][p]=0;
			}
			
		}
		fprintf(OP_1,"\n");
		
		//Calculation for Output of Output layer
		for(p=1;p<=P-5;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<M+1;j++)
				{
					IO[k][p]=IO[k][p]+OH[j][p]*W[j][k];
				}
				IO[k][p]=IO[k][p]+1.0;
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				IO[k][p]=0;
			}
		}
		fprintf(OP_1,"\n");
		
		//Calculations del_W_jk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				del_W[j][k]=0;
				for(p=1;p<=P-5;p++)
				{
					del_W[j][k]=del_W[j][k]+((0.5/P)*(TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*OH[j][p]);
				}
				fprintf(OP_1,"del_W[%d][%d]=%f\t",j,k,del_W[j][k]);
			}
			fprintf(OP_1,"\n");
		}

		fprintf(OP_1,"\n");
		
		//Calcualtions del_V_ij
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				del_V[i][j]=0;
				for(p=1;p<=P-5;p++)
				{
					for(k=1;k<=N;k++)
					{
						del_V[i][j]=del_V[i][j]+((0.5/(P*N))*((TO[k][p]-OO[k][p])*(1-(OO[k][p]*OO[k][p]))*W[j][k]*OH[j][p]*(1-OH[j][p])*I[i][p]));
					}
				}
				fprintf(OP_1,"del_V[%d][%d]=%f\t",i,j,del_V[i][j]);
			}
			fprintf(OP_1,"\n");
		}
		
		//Calcualtion for e
		Mean_sq_E=0;
		for(p=1;p<=P-5;p++)
		{
			for(k=1;k<=N;k++)
			{
				e=pow((TO[k][p]-OO[k][p]),2)/2;
				Mean_sq_E=Mean_sq_E+e;
			}
		}
		Mean_sq_E=Mean_sq_E/P;
		fprintf(OP_1,"\nMean_sq_E=%f\tIteration=%d\n",Mean_sq_E,C);
		
		//Updating values of Vij
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j]=V[i][j]+del_V[i][j];
				fprintf(OP_1,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}

			fprintf(OP_1,"\n");
		}
		fprintf(OP_1,"\n");
		
		//Updating values of Wjk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k]=W[j][k]+del_W[j][k];
				fprintf(OP_1,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			fprintf(OP_1,"\n");
		}
		
	printf("\nIteration %d completed",C);
	
	C++;

	}while(Mean_sq_E>Max_E);

	for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				fprintf(OP_2,"V[%d][%d]:%f\t",i,j,V[i][j]);
			}
			fprintf(OP_2,"\n");
		}
		fprintf(OP_2,"\n");
		
		//Updating values of Wjk
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				fprintf(OP_2,"W[%d][%d]:%f\t",j,k,W[j][k]);
			}
			fprintf(OP_2,"\n");
		}
		
	//Pattern Testing

		//Calculation for forward pass
		
		fprintf(OP_2,"\nForward Pass Calculation Result:");

		for(p=36;p<=40;p++)
		{
			IH[j][p]=0;
			for(j=1;j<M+1;j++)
			{
				for(i=1;i<L+1;i++)
				{
					IH[j][p]=IH[j][p]+(I[i][p]*V[i][j]);
				}
				IH[j][p]=IH[j][p]+1.0;
				OH[j][p]=1/(1+exp(-IH[j][p]));
				fprintf(OP_2,"\nIH[%d][%d]:%f\tOH[%d][%d]:%f",j,p,IH[j][p],j,p,OH[j][p]);
				IH[j][p]=0;
			}
			
		}

		fprintf(OP_2,"\n\n");
		
		fprintf(OP_2,"\nOutput of Output layer:");
		
		//Calculation for Output of Output layer
		for(p=36;p<=40;p++)
		{
			IO[k][p]=0;
			for(k=1;k<N+1;k++)
			{
				for(j=1;j<=M+1;j++)
				{
					IO[k][p]=IO[k][p]+OH[j][p]*W[j][k];
				}
				IO[k][p]=IO[k][p]+1.0;
				OO[k][p]=(exp(IO[k][p])-exp(-1*IO[k][p]))/(exp(IO[k][p])+exp(-1*IO[k][p]));
				fprintf(OP_2,"\nIO[%d][%d]:%f\tOO[%d][%d]:%f\tTO[%d][%d]%f\terror=:%f",k,p,IO[k][p],k,p,OO[k][p],k,p,TO[k][p],fabs(OO[k][p]-TO[k][p]));
				IO[k][p]=0;
			}
		}
	
	fclose(IP);
	fclose(OP_1);
	fclose(OP_2);

	return 0;
}
