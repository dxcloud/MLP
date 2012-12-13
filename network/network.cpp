#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

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
  for( i = 1; i < nbLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nbNodes; j++ )
	{
	  /* --- calcul de la somme pondérée en entrée */
	  float sum = 0.0;
	  for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) 
	    {
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
  for( i = 0; i < pLayers[nbLayers-1].nbNodes; i++) 
    {
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
  for( i = (nbLayers-2); i >= 0; i-- )
    {
      /* --- couche inférieure */
      for( j = 0; j < pLayers[i].nbNodes; j++ )
	{
	  float x = pLayers[i].pNodes[j].x;
	  float E = 0.0;
	  /* --- couche supérieure */
	  for ( k = 0; k < pLayers[i+1].nbNodes; k++ ) 
	    {
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
  for( i = 1; i < nbLayers; i++ )
    {
      for( j = 0; j < pLayers[i].nbNodes; j++ )
	{
	  for ( k = 0; k < pLayers[i-1].nbNodes; k++ ) 
	    {
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

//~ /** 
 //~ * \struct Raster
 //~ * Structure d�crivant une image au format Sun Raster
 //~ */
//~ 
//~ typedef struct {
  //~ struct rasterfile file;  ///< Ent�te image Sun Raster
  //~ unsigned char rouge[256],vert[256],bleu[256];  ///< Palette de couleur
  //~ unsigned char *data;    ///< Pointeur vers l'image
//~ } Raster;
//~ 
//~ 
//~ /**
 //~ * \brief Lecture d'une image au format Sun RASTERFILE.
 //~ *
 //~ * Au retour de cette fonction, la structure r est remplie
 //~ * avec les donn�es li�e � l'image. Le champ r.file contient
 //~ * les informations de l'entete de l'image (dimension, codage, etc).
 //~ * Le champ r.data est un pointeur, allou� par la fonction
 //~ * lire_rasterfile() et qui contient l'image. Cette espace doit
 //~ * �tre lib�r� apr�s usage.
 //~ *
 //~ * \param nom nom du fichier image
 //~ * \param r structure Raster qui contient l'image
 //~ *  charg�e en m�moire
 //~ */
//~ 
//~ void lire_rasterfile(char *nom, Raster *r) {
  //~ FILE *f;
  //~ int i,h,w,w2;
    //~ 
  //~ if( (f=fopen( nom, "r"))==NULL) {
    //~ fprintf(stderr,"erreur a la lecture du fichier %s\n", nom);
    //~ exit(1);
  //~ }
  //~ if (fread( &(r->file), sizeof(struct rasterfile), 1, f) < 1){
    //~ fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  //~ };    
  //~ swap(&(r->file.ras_magic));
  //~ swap(&(r->file.ras_width));
  //~ swap(&(r->file.ras_height));
  //~ swap(&(r->file.ras_depth));
  //~ swap(&(r->file.ras_length));
  //~ swap(&(r->file.ras_type));
  //~ swap(&(r->file.ras_maptype));
  //~ swap(&(r->file.ras_maplength));
    //~ 
  //~ if ((r->file.ras_depth != 8) ||  (r->file.ras_type != RT_STANDARD) ||
      //~ (r->file.ras_maptype != RMT_EQUAL_RGB)) {
    //~ fprintf(stderr,"palette non adaptee\n");
    //~ exit(1);
  //~ }
    //~ 
  //~ /* composante de la palette */
  //~ if (fread(&(r->rouge),r->file.ras_maplength/3,1,f) < 1){ 
    //~ fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  //~ };    
  //~ if (fread(&(r->vert), r->file.ras_maplength/3,1,f) < 1){
    //~ fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  //~ };    
  //~ if (fread(&(r->bleu), r->file.ras_maplength/3,1,f) < 1){
    //~ fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
  //~ };    
    //~ 
  //~ if ((r->data=malloc(r->file.ras_width*r->file.ras_height))==NULL){
    //~ fprintf(stderr,"erreur allocation memoire\n");
    //~ exit(1);
  //~ }
//~ 
  //~ /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   //~ * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
  //~ h=r->file.ras_height;
  //~ w=r->file.ras_width;
  //~ w2=((w + 1) & ~1); /* multiple of 2 greater or equal */
  //~ //  printf("Dans lire_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  //~ for (i=0; i<h; i++){
    //~ if (fread(r->data+i*w,w,1,f) < 1){
      //~ fprintf(stderr, "Error in fread() ar %s:%d\n", __FILE__, __LINE__); 
    //~ }
    //~ if (w2-w > 0){ fseek(f, w2-w, SEEK_CUR); } 
  //~ }
//~ 
  //~ fclose(f);
//~ }
//~ 
//~ /**
 //~ * Sauve une image au format Sun Rasterfile
 //~ */
//~ 
//~ void sauve_rasterfile(char *nom, Raster *r)     {
  //~ FILE *f;
  //~ int i,h,w,w2;
  //~ 
  //~ if( (f=fopen( nom, "w"))==NULL) {
    //~ fprintf(stderr,"erreur a l'ecriture du fichier %s\n", nom);
    //~ exit(1);
  //~ }
    //~ 
  //~ swap(&(r->file.ras_magic));
  //~ swap(&(r->file.ras_width));
  //~ swap(&(r->file.ras_height));
  //~ swap(&(r->file.ras_depth));
  //~ swap(&(r->file.ras_length));
  //~ swap(&(r->file.ras_type));
  //~ swap(&(r->file.ras_maptype));
  //~ swap(&(r->file.ras_maplength));
    //~ 
  //~ fwrite(&(r->file),sizeof(struct rasterfile),1,f);
  //~ /* composante de la palette */
  //~ fwrite(&(r->rouge),256,1,f);
  //~ fwrite(&(r->vert),256,1,f);
  //~ fwrite(&(r->bleu),256,1,f);
  //~ /* pour le reconvertir pour la taille de l'image */
  //~ swap(&(r->file.ras_width));
  //~ swap(&(r->file.ras_height));
//~ 
  //~ /* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
   //~ * (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
  //~ h=r->file.ras_height;
  //~ w=r->file.ras_width;
  //~ w2=((w + 1) & ~1); /* multiple of 2 greater or equal */
  //~ //  printf("Dans lire_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
  //~ for (i=0; i<h; i++){
    //~ if (fwrite(r->data+i*w,w,1,f) < 1){
      //~ fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__); 
    //~ }
    //~ if (w2-w > 0){ /* padding */
      //~ unsigned char zeros[1]={0}; 
      //~ if (w2-w != 1){ fprintf(stderr, "Error in sauve_rasterfile(): w2-w != 1 \n"); }
      //~ if (fwrite(zeros, w2-w, 1, f) < 1){
	//~ fprintf(stderr, "Error in fwrite() ar %s:%d\n", __FILE__, __LINE__); 
      //~ }
    //~ } 
  //~ }
//~ 
  //~ fclose(f);
//~ }
//~ 
//~ 
//~ 
//~ 
//~ /**
 //~ * Conversion d'une image avec un "unsigned char" par pixel en une image 
 //~ * avec un "float" par pixel. 
 //~ */
//~ 
//~ void convert_uchar2float_image(unsigned char*p_ua, float *p_f, int h, int w){
  //~ int i,j;
  //~ 
  //~ for (i=0; i<h; i++){
    //~ for(j=0; j<w; j++){
      //~ p_f[i*w+j] = (float) p_ua[i*w+j]; 
    //~ }
  //~ }
//~ }
//~ 
//~ 
//~ 
//~ /**
 //~ * Conversion d'une image avec un "float" par pixel en une image 
 //~ * avec un "unsigned char" par pixel. 
 //~ */
//~ 
//~ void convert_float2uchar_image(float *p_f, unsigned char*p_ua, int h, int w){
  //~ int i,j;
  //~ 
  //~ for (i=0; i<h; i++){
    //~ for(j=0; j<w; j++){
      //~ p_ua[i*w+j] = (unsigned char) rintf(p_f[i*w+j]); 
    //~ }
  //~ }
//~ }


