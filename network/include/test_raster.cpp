#include "rasterfile.hpp"
#include <string>

int main(int argc, char *argv[])
{
	if (argc != 2) return 1;
	std::string file_name = argv[1];
	Raster r;
	if (r.read_raster(file_name) != true)
		return 1;
	for (int i = 0; i < r.width()*r.height(); ++i) r[i] /= 2;
	if (r.save_raster("new_image.ras"))
		return 1;
	
	return 0;
}
