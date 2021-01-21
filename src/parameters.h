#ifndef parameters_h
#define parameters_h
#include <petsc.h>

typedef enum {
    STROUHAL = 0,
    REYNOLDS,
    FROUDE,
    PECLET,
    HEATRELEASE,
    GAMMA,
    PTH,
    MU,
    K,
    CP,
    BETA,
    TOTAlCONSTANTS
} FlowConstants;

typedef struct {
    PetscReal strouhal;
    PetscReal reynolds;
    PetscReal froude;
    PetscReal peclet;
    PetscReal heatRelease;
    PetscReal gamma;
    PetscReal pth;   /* non-dimensional constant thermodynamic pressure */
    PetscReal mu;    /* non-dimensional viscosity */
    PetscReal k;     /* non-dimensional thermal conductivity */
    PetscReal cp;    /* non-dimensional specific heat capacity */
    PetscReal beta;  /* non-dimensional thermal expansion coefficient */
} FlowParameters;

PETSC_EXTERN void PackFlowParameters(FlowParameters *parameters, PetscScalar *constantArray);
PETSC_EXTERN PetscErrorCode SetupFlowParameters(PetscBag *flowParametersBag);

#endif