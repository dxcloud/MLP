#include "rasterfile.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>

/**
 * @fn void swap(int *i)
 * @brief Cette procedure convertit un entier LINUX en un entier SUN
 * @param i Pointeur vers l'entier a convertir
 */
void swap(int *i) {
	unsigned char s[4], *n;
	memcpy(s, i, 4);
	n = (unsigned char *) i;
	n[0] = s[3];
	n[1] = s[2];
	n[2] = s[1];
	n[3] = s[0];
}

bool Raster::read_raster(const std::string& file_name)
{
	FILE * f;
	int i, h, w, w2;

	//~ std::ifstream f(file_name.c_str(), std::ios::in);
	if ((f = fopen(file_name.c_str(), "r")) == NULL) {
		std::cerr << "Error in read_raster(): " << file_name << std::endl;
		return false;
	}

	if (fread(&file, sizeof(Rasterfile), 1, f) < 1) {
		std::cerr << "Error in fread() at " << __FILE__ << ":" << __LINE__ << std::endl;
		return false;
	}
	swap(&(file.ras_magic));
	swap(&(file.ras_width));
	swap(&(file.ras_height));
	swap(&(file.ras_depth));
	swap(&(file.ras_length));
	swap(&(file.ras_type));
	swap(&(file.ras_maptype));
	swap(&(file.ras_maplength));

	if ((file.ras_depth != 8) ||  (file.ras_type != RT_STANDARD) ||
		(file.ras_maptype != RMT_EQUAL_RGB)) {
		std::cerr << "palette non adaptee" << std::endl;
		std::cout << file.ras_depth << "," << file.ras_type << "," << file.ras_maptype << std::endl;
		return false;
	}

		/* composante de la palette */
	if (fread(&red, file.ras_maplength / 3, 1, f) < 1) {
		std::cerr << "Error in fread() at " << __FILE__ << ":" << __LINE__ << std::endl;
		return false;
	}
	if (fread(&green, file.ras_maplength / 3, 1, f) < 1) {
		std::cerr << "Error in fread() at " << __FILE__ << ":" << __LINE__ << std::endl;
		return false;
	}
	if (fread(&blue, file.ras_maplength / 3, 1, f) < 1) {
		std::cerr << "Error in fread() at " << __FILE__ << ":" << __LINE__ << std::endl;
		return false;
	};

	if ((data = new unsigned char [file.ras_width * file.ras_height]) == NULL) {
		std::cerr << "erreur allocation memoire" << std::endl;
		return false;
	}
		
		/* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
		* (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
	h = file.ras_height;
	w = file.ras_width;
	w2 = ((w + 1) & ~1); /* multiple of 2 greater or equal */
	//  printf("Dans lire_rasterfile(): h=%d w=%d w2=%d\n", h, w, w2);
	for (i = 0; i < h; i++){
		if (fread(data + i * w, w, 1, f) < 1) {
			std::cerr << "Error in fread() at " << __FILE__ << ":" << __LINE__ << std::endl;
		}
		if (w2 - w > 0) { fseek(f, w2 - w, SEEK_CUR); }
	}
	fclose(f);
	return true;
}

bool Raster::save_raster(const std::string& file_name)
{
	FILE *f;
	int i, h, w, w2;

	if((f = fopen(file_name.c_str(), "w")) == NULL) {
		std::cerr << "Erreur ecriture fichier: " << std::endl;
		return false;
	}

	swap(&(file.ras_magic));
	swap(&(file.ras_width));
	swap(&(file.ras_height));
	swap(&(file.ras_depth));
	swap(&(file.ras_length));
	swap(&(file.ras_type));
	swap(&(file.ras_maptype));
	swap(&(file.ras_maplength));

	fwrite(&file, sizeof(Rasterfile), 1, f);
/* composante de la palette */
	fwrite(&red, 256, 1, f);
	fwrite(&green, 256, 1, f);
	fwrite(&blue, 256, 1, f);
/* pour le reconvertir pour la taille de l'image */
	swap(&(file.ras_width));
	swap(&(file.ras_height));

/* Format Sun Rasterfile: "The width of a scan line is always a multiple of 16 bits, padded when necessary."
* (see: http://netghost.narod.ru/gff/graphics/summary/sunras.htm) */ 
	h = file.ras_height;
	w = file.ras_width;
	w2 = ((w + 1) & ~1); /* multiple of 2 greater or equal */
	for (i = 0; i < h; i++) {
		if (fwrite(data + i * w, w, 1, f) < 1) {
			std::cerr << "Error in fwrite() at " << __FILE__ << ":" << __LINE__ << std::endl;
		}
		if (w2 - w > 0){ /* padding */
			unsigned char zeros[1] = {0};
			if (w2 - w != 1){
				std::cerr << "Error in in save_raster()" << std::endl;
			}
			if (fwrite(zeros, w2 - w, 1, f) < 1) {
				std::cerr << "Error in fwrite() at " << __FILE__ << ":" << __LINE__ << std::endl;
			}
		} 
	}
	fclose(f);
	return true;
}
