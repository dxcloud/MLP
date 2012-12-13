/* -------------------------------------- */
/* Retro propagation du gradient          */
/*	Reseau multi-couches              */
/*				  	  */
/* Auteur: Arrouy William                 */
/* Date  : Fevrier 1996                   */
/* Version : 1.1                          */
/* -------------------------------------- */

/* Fichiers d'entetes */
/* ------------------ */

#include<stdio.h>
#include<stdlib.h>
#include<math.h> 
#include<time.h>
 
/* Structures */
/* ---------- */

struct neurone
{
	float *w;					
	float sortie;
	float erreur;
};

/* Constantes du reseau */
/* -------------------- */

#define CACHE_MAX	10			/* Nombre maximal de couches cachees */
#define SEUIL_APP	0.05		/* Erreur moyenne toleree en sortie */
#define SEUIL		0.5		/* Seuil pour vrai/faux */

const long MAX_ITER = 100000;  /* Nombre d'iterations en apprentissage */

#undef	GENERALISATION

/* Debut des procedures */
/* ++++++++++++++++++++ */

/* Initialisation des parametres du reseau */
/* --------------------------------------- */

void init_reseau(int *nb_neurones, int *nb_cache)
{
	int i;

	printf("Retropropagation du gradient \n\n");

	printf("Nombre de neurones d'entree		: ");
	scanf("%d", &nb_neurones[0]);

	printf("Nombre de couches intermediaires	: ");
	scanf("%d", nb_cache);

	if (*nb_cache > CACHE_MAX)
		*nb_cache = CACHE_MAX;

	for(i=0;i<*nb_cache;i++)
	{
		printf("  Nombre de neurones couche cachee No %d : ", i+1);
		scanf("%d", &nb_neurones[i+1]);
	}
	
	printf("Nombre de neurones de sortie		: ");
	scanf("%d", &nb_neurones[*nb_cache+1]);
}

/* Allocation du reseau */
/* -------------------- */

int alloc_reseau(struct neurone ***n, int *nb_neurones, int nb_cache)
{
	struct neurone **n1;
 	int i, j;
	
	/* Allocation des couches */

	if ( (n1 = (struct neurone **) malloc (sizeof(struct neurone*) * (nb_cache+2))) == NULL)
		return -1;
	
	/* Allocation des neurones */

	for(i=0;i<nb_cache+2;i++)
		if ( (n1[i] = (struct neurone *) malloc (sizeof (struct neurone) * nb_neurones[i]) ) == NULL )
 		{
			free(n1);
			return -1;
		}

	/* Allocation des poids */
	
	for(i=0;i<nb_cache+1;i++)
		for(j=0;j<nb_neurones[i];j++)
			if ( (n1[i][j].w = (float *) malloc(sizeof(float) * nb_neurones[i+1]) ) == NULL )
			{
				free(n1);
				return -1;
			}

	*n = n1;
	return 0;
}

/* Fonctions d'activation */
/* ++++++++++++++++++++++ */

/* Fonction sigmoide */
/* ----------------- */

float sigmoide(float valeur)
{
	if (fabs(valeur)<1e-10)
		return 0.5;
	else
		return ( 1.0 / ( 1.0 + exp(-valeur) ) );
}

/* Fonction derivee de la sigmoide */
/* ------------------------------- */

float derivee(float valeur)
{
	return ( valeur * (1 - valeur) );
}

/* Fonctions generales */
/* ------------------- */

void init_poids(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j, k;
	int t;
	
	/* Initialisation du generateur aleatoire */

	time((time_t *)&t);
	srand(t);

	/* Sur toutes les couches sauf la derniere */
		/* Sur tous les neurones de la couche */
			/*Pour chaque poids */

	for(i=0;i<nb_cache+1;i++)
		for(j=0;j<nb_neurones[i];j++)
			for(k=0;k<nb_neurones[i+1];k++)
				n[i][j].w[k] = 1.0 * rand() / RAND_MAX * 1.0;
}

/* Apprentissage */
/* +++++++++++++ */

/* Active entree */
/* ------------- */

void active_entree(struct neurone **n, int *exemple, int nb_entree)
{
	int i;

	/* Pour tous les neurones de la couche d'entree */

	for(i=0;i<nb_entree;i++)
		n[0][i].sortie = 1.0 * exemple[i];
}

/* Propagation vers l'avant */
/* ------------------------ */

void propage_vers_l_avant(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j, k;
	float valeur;

	for(i=1;i<nb_cache+2;i++)
	{
		for(j=0;j<nb_neurones[i];j++)
		{
			valeur = 0.0;

			/* Somme sur toute les liaisons */

			for(k=0;k<nb_neurones[i-1];k++)
				valeur+=n[i-1][k].w[j] * n[i-1][k].sortie;

			n[i][j].sortie = sigmoide(valeur);
		}
	}
}

/* Calcul de l'erreur finale */
/* ------------------------- */

float calcul_erreur_finale(struct neurone **n, int *exemple, int *nb_neurones, int nb_cache)
{
	int i;
	float erreur_sortie = 0.0;
	float erreur_brute;

	/* Sur tous les neurones de sortie */

	for(i=0;i<nb_neurones[nb_cache+1];i++)
	{
		erreur_brute = (1.0 * exemple[nb_neurones[0]+i]) - n[nb_cache+1][i].sortie;
		n[nb_cache+1][i].erreur = erreur_brute * derivee(n[nb_cache+1][i].sortie);
		erreur_sortie += 0.5 * erreur_brute * erreur_brute; 	
	}		
	
	return erreur_sortie;
}

/* Propagation de l'erreur vers l'arriere */
/* -------------------------------------- */

void propage_erreur_arriere(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j, k;
	float erreur;
	
	/* Pour tous les neurones de la couche intermediaire */

	for(i=nb_cache;i>0;i--)
	{
		for(j=0;j<nb_neurones[i];j++)
		{
			erreur = 0;
			
			/* pour toutes les liaisons */

			for(k=0;k<nb_neurones[i+1];k++)
				erreur+= n[i+1][k].erreur * n[i][j].w[k];
				n[i][j].erreur = erreur * derivee(n[i][j].sortie);
		}
	}
}

/* Ajustement des poids */
/* -------------------- */


void ajuste_poids(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j, k;

	/* Pour toutes les couches sauf celle de sortie */

	for(i=nb_cache;i>=0;i--)
		for(j=0;j<nb_neurones[i];j++)
			for(k=0;k<nb_neurones[i+1];k++)
				n[i][j].w[k]+= n[i+1][k].erreur * n[i][j].sortie;

				/* Poids += erreur dest * sortie src */
}
 
/* Lancement du reseau */
/* ------------------- */

float calcule(struct neurone **n, int *exemple, int *nb_neurones, int nb_cache)
{
	float erreur;

 	active_entree(n, exemple, nb_neurones[0]);
	propage_vers_l_avant(n, nb_neurones, nb_cache);
	erreur = calcul_erreur_finale(n, exemple, nb_neurones, nb_cache);
	propage_erreur_arriere(n, nb_neurones, nb_cache);
	ajuste_poids(n, nb_neurones, nb_cache);
	
	return erreur;
}

/* Fonctions de lecture */
/* -------------------- */

int lecture_exemples(int ***e, int *nb_app, int total)
{
	FILE *fp;
	char buf[30];
	int **e1;

	do
	{ 
		printf("\n");
		printf("Nom du fichier d'apprentissage : ");
		scanf("%s", buf);

		if ( (fp = fopen(buf, "r+") ) == NULL )
		{
			printf("  Impossible d'ouvrir le fichier\n");
		}
		else
		{
			int i, j;

			fscanf(fp,"%d", nb_app);
		
			/* Allocation des exemples */

			if ( (e1 = (int **) malloc(sizeof(int *) * (*nb_app))) == NULL)
			{
				fclose(fp);
				return -1;
			}	
			
			for(i=0;i<*nb_app;i++)
				if( (e1[i] = (int *) malloc(sizeof(int) * total)) == NULL)
				{
					fclose(fp);
					return -1;
				}
			
			/* Lecture des exemples */

			for(i=0;i<*nb_app;i++)
				for(j=0;j<total;j++)
					fscanf(fp, "%d", &e1[i][j]);
		}
	}
	while(fp==NULL);
	
	fclose(fp);
	*e = e1;
	return 0;
}

/* Fonctions de verification */
/* +++++++++++++++++++++++++ */

/* Affichage des poids */
/* ------------------- */

void affiche_poids(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j, k;
	
	for(i=0;i<nb_cache+1;i++)
	{
		printf("Couche No %d \n", i);
		for(j=0;j<nb_neurones[i];j++)
		{
			printf("  Neurone No %d \n    ", j+1);
			for(k=0;k<nb_neurones[i+1];k++)
				printf(" %2.4f ",n[i][j].w[k]);
			printf("\n");
		}
	}
}

/* Affichage des activations */
/* ------------------------- */

void affiche_sortie(struct neurone **n, int *nb_neurones, int nb_cache)
{
	int i, j;
	
	for(i=0;i<nb_cache+2;i++)
	{
		printf("Couche No %d \n", i);
		for(j=0;j<nb_neurones[i];j++)
		{
			printf("     Neurone No %d   %f\n", j+1, n[i][j].sortie);
		}
	}
}

/* Lancement de l'apprentissage */
/* ---------------------------- */

void lance_apprentissage(struct neurone **n, int **exemples, int *nb_neurones, int nb_cache, int nb_app)
{
	int i;
	float erreur;
 	long iter = 0;	

	do
	{
		iter++;
		erreur = 0.0;
	 
 		for(i=0;i<nb_app;i++)
			erreur += calcule(n, exemples[i], nb_neurones, nb_cache);
	}
	while( (erreur>SEUIL_APP * nb_app) && (iter<(long)MAX_ITER) );
  	
	printf("\nPhase Apprentissage\n\n");
	printf(" Convergence en %d iterations pour ecart moyen effectif %f (%f) \n", iter, SEUIL_APP, erreur/nb_app);
	printf("  Ecart moyen %f pour %d exemples\n", erreur, nb_app);
}

/* Verification et calcul des sorties */
/* ++++++++++++++++++++++++++++++++++ */

/* Test sur les exemples */
/* --------------------- */

void general_sur_exemple(struct neurone **n, int **exemples, int *nb_neurones, int nb_cache, int nb_app)
{
	int i;
	int faux = 0;

	printf("\nPhase de generalisation sur apprentissage\n\n");

	/* Pour chaque exemple calcule la sortie */
	
	for(i=0;i<nb_app;i++)
	{
		int sortie;
		int err = 0;
		int j;

 		active_entree(n, exemples[i], nb_neurones[0]);
		propage_vers_l_avant(n, nb_neurones, nb_cache);
		
		/* Verifie la sortie reelle et attendue */

		for(j=0;j<nb_neurones[nb_cache+1];j++)
		{
			if (n[nb_cache+1][j].sortie >= SEUIL)
				sortie = 1;
			else
				sortie = 0;
			
			if (exemples[i][nb_neurones[0]+j] != sortie)
				err = 1;
		}
		faux += err;
	}

	printf("  Erreur %d sur %d soit %3.2f \n", faux, nb_app, (faux * 100.0) / (nb_app * 1.0)); 
}

/* Generalisation */
/* -------------- */

void generalisation(struct neurone **n, int *exemple, int *nb_neurones, int nb_cache)
{
	FILE *fp;
	char buf[30];

	do
	{
		printf("\n");
		printf("Nom du fichier d'apprentissage : ");
		scanf("%s", buf);

		if ( (fp = fopen(buf, "r+") ) == NULL )
		{
			printf("  Impossible d'ouvrir le fichier\n");
		}
		else
		{
			int i, j;
			int nb_gen;
			int faux = 0;

			printf("\nPhase de generalisation\n\n");

			fscanf(fp,"%d", &nb_gen);
		 
			for(i=0;i<nb_gen;i++)
			{
				int sortie;
				int err = 0;
				int k;

				/* Lecture de l'exemple */

				for(j=0;j<nb_neurones[0]+nb_neurones[nb_cache+1];j++)
					fscanf(fp, "%d", &exemple[j]);
	
 				active_entree(n, exemple, nb_neurones[0]);
				propage_vers_l_avant(n, nb_neurones, nb_cache);
		
				/* Verifie la sortie reelle et attendue */

				for(k=0;k<nb_neurones[nb_cache+1];k++)
				{ 
					if (n[nb_cache+1][k].sortie >= SEUIL)
						sortie = 1;
					else
						sortie = 0;
			
					if (exemple[nb_neurones[0]+k] != sortie)
						err = 1;
				}
				faux += err;
			}
			printf("  Erreur %d sur %d soit %3.2f \n", faux, nb_gen, (faux * 100.0) / (nb_gen * 1.0)); 
		}
	}     
	while(fp==NULL);
}

/* Point d'entree */
/* -------------- */

int main()
{
	int nb_cache;			/* Nombre de couches intermediaires */
	int nb_neurones[CACHE_MAX+2];	/* Nombre de neurones */
	struct neurone **neurones;	
	int **exemples;			/* Exemplaires */
	int nb_app;			/* Nombre de tests */

 	init_reseau(nb_neurones, &nb_cache);

	if ( alloc_reseau(&neurones, nb_neurones, nb_cache) == -1 )
	{
		printf("Allocation impossible\n");
		exit(-1);
	}

	init_poids(neurones, nb_neurones, nb_cache);

	if (lecture_exemples(&exemples, &nb_app, nb_neurones[0]+nb_neurones[nb_cache+1]) == -1 )
	{
		printf("Lecture impossible\n");
		exit(-1);
	}

	lance_apprentissage(neurones, exemples, nb_neurones, nb_cache, nb_app);

	general_sur_exemple(neurones, exemples, nb_neurones, nb_cache, nb_app);

#ifdef GENERALISATION
	generalisation(neurones, exemples[0], nb_neurones, nb_cache);
#endif

	/* affiche_poids(neurones, nb_neurones, nb_cache);  */
	return 0;
}

