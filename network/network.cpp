#include <iostream>
#include <stdio.h>
//#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>

#include "rasterfile.h"
#include "network.hpp"

void InitializeRandoms()
{
  //  srand( (unsigned)time( NULL ) );
  srand(4711);
}

int RandomEqualINT(int Low, int High)
{
  return rand() % (High-Low+1) + Low;
}

float RandomEqualREAL(float Low, float High)
{
  return ((float) rand() / RAND_MAX) * (High-Low) + Low;
}

Network::Network(int nl, int npl[]) : 
  nbLayers(0),
  pLayers(0),
  dEta(0.25),
  dAlpha(0.9),
  dGain(1.0),
  dAvgTestError(0.0),
  dMSE(0.0),
  dMAE(0.0)
{
  int i,j;
  
  /* --- création des couches */
  nbLayers = nl;
  pLayers    = new Layer[nl];

  /* --- init des couches */
  for ( i = 0; i < nl; i++ ) 
    {

      /* --- création des Nodees */
      pLayers[i].nbNodes = npl[i];
      pLayers[i].pNodes    = new Node[ npl[i] ];

      /* --- init des Nodees */
      for( j = 0; j < npl[i]; j++ )
	{
	  pLayers[i].pNodes[j].x  = 1.0;
	  pLayers[i].pNodes[j].e  = 0.0;
	  if(i>0)
	    {
	      pLayers[i].pNodes[j].w     = new float[ npl[i-1] ];
	      pLayers[i].pNodes[j].dw    = new float[ npl[i-1] ];
	      pLayers[i].pNodes[j].wsave = new float[ npl[i-1] ];
	    }
	  else
	    {
	      pLayers[i].pNodes[j].w     = NULL;
	      pLayers[i].pNodes[j].dw    = NULL;
	      pLayers[i].pNodes[j].wsave = NULL;
	    }
	}

    }

  
}

Network::~Network()
{
  int i,j;
  for( i = 0; i < nbLayers; i++ )
    {
      if ( pLayers[i].pNodes ) 
	{
	  for( j = 0; j < pLayers[i].nbNodes; j++ )
	    {
	      if ( pLayers[i].pNodes[j].w )
		delete[] pLayers[i].pNodes[j].w;
	      if ( pLayers[i].pNodes[j].dw )
		delete[] pLayers[i].pNodes[j].dw;
	      if ( pLayers[i].pNodes[j].wsave )
		delete[] pLayers[i].pNodes[j].wsave;
	    }
	}
      delete[] pLayers[i].pNodes;
    }
  delete[] pLayers;
}

void Network::randomWeights()
{
  int i,j,k;
  for( i = 1; i < nbLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nbNodes; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) 
	    {
	      pLayers[i].pNodes[j].w [k]    = RandomEqualREAL(-0.5, 0.5);
	      pLayers[i].pNodes[j].dw[k]    = 0.0;
	      pLayers[i].pNodes[j].wsave[k] = 0.0;
	    }
	}
    }
}

void Network::SetInputSignal(float* input)
{
  int i;
  for ( i = 0; i < pLayers[0].nbNodes; i++ )
    {
      pLayers[0].pNodes[i].x = input[i];      
    }
}

void Network::GetOutputSignal(float* output)
{
  int i;
  for ( i = 0; i < pLayers[nbLayers-1].nbNodes; i++ )
    {
      output[i] = pLayers[nbLayers-1].pNodes[i].x;
    }
}

void Network::SaveWeights()
{
  int i,j,k;
  for( i = 1; i < nbLayers; i++ )
    for( j = 0; j < pLayers[i].nbNodes; j++ )
      for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) 
	pLayers[i].pNodes[j].wsave[k] = pLayers[i].pNodes[j].w[k];
}

void Network::RestoreWeights()
{
  int i,j,k;
  for( i = 1; i < nbLayers; i++ )
    for( j = 0; j < pLayers[i].nbNodes; j++ )
      for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) 
	pLayers[i].pNodes[j].w[k] = pLayers[i].pNodes[j].wsave[k];
}

/***************************************************************************/
/* calculate and feedforward outputs from the first layer to the last      */
void Network::PropagateSignal()
{
	int i,j,k;

  /* --- la boucle commence avec la seconde couche */
	for( i = 1; i < nbLayers; i++ ) {
		for( j = 0; j < pLayers[i].nbNodes; j++ ) {
	  /* --- calcul de la somme pondérée en entrée */
			float sum = 0.0;
			for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) {
				float out = pLayers[i-1].pNodes[k].x;
				float w   = pLayers[i  ].pNodes[j].w[k];
				sum += w * out;
			}
	  /* --- application de la fonction d'activation (sigmoid) */
			pLayers[i].pNodes[j].x = 1.0 / (1.0 + exp(-dGain * sum));
		}
	}
}

void Network::ComputeOutputError(float* target)
{
	int  i;
	dMSE = 0.0;
	dMAE = 0.0;
	for( i = 0; i < pLayers[nbLayers-1].nbNodes; i++) {
		float x = pLayers[nbLayers-1].pNodes[i].x;
		float d = target[i] - x;
		pLayers[nbLayers-1].pNodes[i].e = dGain * x * (1.0 - x) * d;
		dMSE += (d * d);
		dMAE += fabs(d);
	}
	/* --- erreur quadratique moyenne */
	dMSE /= (float)pLayers[nbLayers-1].nbNodes;
	/* --- erreur absolue moyenne */
	dMAE /= (float)pLayers[nbLayers-1].nbNodes;
}

/***************************************************************************/
/* backpropagate error from the output layer through to the first layer    */

void Network::BackPropagateError()
{
	int i,j,k;
	/* --- la boucle commence à l'avant dernière couche */
	for( i = (nbLayers-2); i >= 0; i-- ) {
	/* --- couche inférieure */
		for( j = 0; j < pLayers[i].nbNodes; j++ ) {
			float x = pLayers[i].pNodes[j].x;
			float E = 0.0;
	/* --- couche supérieure */
			for ( k = 0; k < pLayers[i+1].nbNodes; k++ ) {
				E += pLayers[i+1].pNodes[k].w[j] * pLayers[i+1].pNodes[k].e;
			}
			pLayers[i].pNodes[j].e = dGain * x * (1.0 - x) * E;
		}
	}
}

/***************************************************************************/
/* update weights for all of the Nodes from the first to the last layer  */

void Network::AdjustWeights()
{
	int i,j,k;
/* --- la boucle commence avec la seconde couche */
	for( i = 1; i < nbLayers; i++ ) {
		for( j = 0; j < pLayers[i].nbNodes; j++ ) {
			for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) {
				float x  = pLayers[i-1].pNodes[k].x;
				float e  = pLayers[i  ].pNodes[j].e;
				float dw = pLayers[i  ].pNodes[j].dw[k];
				pLayers[i].pNodes[j].w [k] += dEta * x * e + dAlpha * dw;
				pLayers[i].pNodes[j].dw[k]  = dEta * x * e;
			}
		}
	}
}

void Network::Simulate(float* input, float* output, float* target, bool training)
{

  if(!input)  return;
  if(!target) return;
  
  /* --- on fait passer le signal dans le réseau */
  SetInputSignal(input);
  PropagateSignal();
  if(output) GetOutputSignal(output);

  if(output && !training) printf("test: %.2f %.2f %.2f = %.2f\n", input[0],input[1],target[0],output[0]);

  /* --- calcul de l'erreur en sortie par rapport à la cible */
  /*     ce calcul sert de base pour la rétropropagation     */
  ComputeOutputError(target);

  /* --- si c'est un apprentissage, on fait une rétropropagation de l'erreur */
  if (training) 
    {
      BackPropagateError();
      AdjustWeights();
    }
}

bool read_number(FILE* fp, float* number)
{
  char szWord[256];
  int i = 0;
  int b;

  *number = 0.0;

  szWord[0] = '\0';
  while ( ((b=fgetc(fp))!=EOF) && (i<255) )
    {
      if( (b=='.') ||
	  (b=='0') ||
	  (b=='1') ||
	  (b=='2') ||
	  (b=='3') ||
	  (b=='4') ||
	  (b=='5') ||
	  (b=='6') ||
	  (b=='7') ||
	  (b=='8') ||
	  (b=='9') )
	{
	  szWord[i++] = (char)b;
	}
      else
	if(i>0) break;
    }
  szWord[i] = '\0';

  if(i==0) return false;

  *number = atof(szWord);

  return true;
}

int Network::Train(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  float* input  = NULL;
  float* output = NULL;
  float* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");
  if(!fp) return 0;

  input  = new float[pLayers[0].nbNodes];
  output = new float[pLayers[nbLayers-1].nbNodes];
  target = new float[pLayers[nbLayers-1].nbNodes];

  if(!input) return 0;
  if(!output) return 0;
  if(!target) return 0;


  while( !feof(fp) )
    {
      float dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on le transforme en input/target */
	  if( nbi < pLayers[0].nbNodes ) 
	    input[nbi++] = dNumber;
	  else if( nbt < pLayers[nbLayers-1].nbNodes )
	    target[nbt++] = dNumber;

	  /* --- on fait un apprentisage du réseau  avec cette ligne*/
	  if( (nbi == pLayers[0].nbNodes) &&
	      (nbt == pLayers[nbLayers-1].nbNodes) ) 
	    {
	      Simulate(input, output, target, true);
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int Network::Test(const char* fname)
{
  int count = 0;
  int nbi   = 0;
  int nbt   = 0;
  float* input  = NULL;
  float* output = NULL;
  float* target = NULL;
  FILE*   fp = NULL;

  fp = fopen(fname,"r");
  if(!fp) return 0;

  input  = new float[pLayers[0].nbNodes];
  output = new float[pLayers[nbLayers-1].nbNodes];
  target = new float[pLayers[nbLayers-1].nbNodes];

  if(!input) return 0;
  if(!output) return 0;
  if(!target) return 0;

  dAvgTestError = 0.0;

  while( !feof(fp) )
    {
      float dNumber;
      if( read_number(fp,&dNumber) )
	{
	  /* --- on le transforme en input/target */
	  if( nbi < pLayers[0].nbNodes ) 
	    input[nbi++] = dNumber;
	  else if( nbt < pLayers[nbLayers-1].nbNodes )
	    target[nbt++] = dNumber;

	  /* --- on fait un apprentisage du réseau  avec cette ligne*/
	  if( (nbi == pLayers[0].nbNodes) &&
	      (nbt == pLayers[nbLayers-1].nbNodes) ) 
	    {
	      Simulate(input, output, target, false);
	      dAvgTestError += dMAE;
	      nbi = 0;
	      nbt = 0;
	      count++;
	    }
	}
      else
	{
	  break;
	}
    }

  dAvgTestError /= count;

  if(fp) fclose(fp);

  if(input)  delete[] input;
  if(output) delete[] output;
  if(target) delete[] target;

  return count;
}

int Network::Evaluate()
{
  int count = 0;
  return count;
}

void Network::Run(const char* fname, const int& maxiter)
{
  int    countTrain = 0;
  int    countLines = 0;
  bool   Stop = false;
  bool   firstIter = true;
  float dMinTestError = 0.0;

  /* --- init du générateur de nombres aléatoires  */
  /* --- et génération des pondérations aléatoires */
  InitializeRandoms();
  randomWeights();

  /* --- on lance l'apprentissage avec tests */
  do {

    countLines += Train(fname);
    Test(fname);
    countTrain++;

    if(firstIter)
      {
	dMinTestError = dAvgTestError;
	firstIter = false;
      }

    printf( "%i \t TestError: %f", countTrain, dAvgTestError);

    if ( dAvgTestError < dMinTestError) 
      {
	printf(" -> saving weights\n");
	dMinTestError = dAvgTestError;
	SaveWeights();
      }
    else if (dAvgTestError > 1.2 * dMinTestError) 
      {
	printf(" -> stopping training and restoring weights\n");
	Stop = true;
	RestoreWeights();
      }
    else
      {
	printf(" -> ok\n");
      }

  } while ( (!Stop) && (countTrain<maxiter) );

  printf("bye\n");

}



