static char help[] = "Time-dependent Low Mach Flow in 2d channels with finite elements.\n\
We solve the Low Mach flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

#include "demo.h"
#include "mesh.h"

int main(int argc,char **args)
{
    int ierr = demo(argc, args);
    return ierr;
}