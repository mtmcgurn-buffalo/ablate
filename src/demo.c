#include "demo.h"

static char help[] = "Simple 2D Transient Head Conduction";

#include <petscbag.h>
#include <petscts.h>
#include <petscdmplex.h>
#include <petscds.h>

static const PetscBool SIMPLEX = PETSC_TRUE;

typedef struct {
    PetscReal rhoCp;
    PetscReal k; /* Thermal diffusivity */
} Parameter;

typedef struct {
    /* Problem definition */
    PetscBag bag;     /* Holds problem parameters */
} AppCtx;


static PetscErrorCode cubic_u(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = time + X[0]*X[0] + X[1]*X[1];
  return 0;
}
static PetscErrorCode cubic_u_t(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 1.0;
  return 0;
}


static void g0_temp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
    g0[0] = u_tShift/(u[0]*u[0]) - 2.0/(u[0]*u[0]*u[0])*u_t[0];
//    g0[0] = -2.0/(u[0]*u[0]*u[0]);
  if(PetscAbsReal(x[0] - 0.589279) < 1E-5 && PetscAbsReal(x[1] - 0.833195) < 1E-5) {
      printf("g0_temp= %f, %f, %f, %f, (%f,%f)\n", u_t[0], g0[0], u[0], t, x[0],
             x[1]);
    }
}

// integrand for test function gradient term
static void f0_heat_conduction(PetscInt dim, PetscInt nf, PetscInt nfAux,
                               const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                               PetscReal t, const PetscReal X[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt    d;

  PetscReal T = t + X[0]*X[0] + X[1]*X[1];

  f0[0] = -1.0/(T*T) * 1.0;
  f0[0] += u_t[0]/(u[0]*u[0]);
  if(PetscAbsReal(X[0] - 0.589279) < 1E-5 && PetscAbsReal(X[1] - 0.833195) < 1E-5) {
    printf("f0_heat_= %f, %f, %f, %f, (%f,%f\n", u_t[0], f0[0], u[0], t, X[0],
           X[1]);
  }

//  f0[0] += 1.0/(u[0]*u[0]);

}

static PetscErrorCode SetupParameters(AppCtx *user)
{
    PetscBag       bag;
    Parameter     *p;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    /* setup PETSc parameter bag */
    ierr = PetscBagGetData(user->bag, (void **) &p);CHKERRQ(ierr);
    ierr = PetscBagSetName(user->bag, "par", "Thermal field parameters");CHKERRQ(ierr);
    bag  = user->bag;
    ierr = PetscBagRegisterReal(bag, &p->k,    1.0, "k",    "thermal conductivity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(bag, &p->rhoCp, 100.0, "rhoCp", "rhoCp");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

//static PetscErrorCode boundary(PetscInt Dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
//{
//    u[0] = 1000;
//    return 0;
//}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, 2, SIMPLEX, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
    PetscErrorCode   ierr;

    PetscFunctionBeginUser;

    // get the discrete system from the domain
    PetscDS ds;
    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);

    // setup the residual for the discrete system
    ierr = PetscDSSetResidual(
            ds,
            0,// the test field number
            f0_heat_conduction, // f0	- integrand for the test function term
            NULL    //f1	- integrand for the test function gradient term
    );CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(ds, 0, 0, g0_temp, NULL, NULL, NULL);CHKERRQ(ierr);
//    ierr = PetscDSSetBdResidual(ds, 0, f0_bd, NULL);CHKERRQ(ierr);


    /* Setup constants */
    {
        Parameter  *param;
        PetscScalar constants[2];

        ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
        constants[0] = param->rhoCp;
        constants[1] = param->k;
        ierr = PetscDSSetConstants(ds, 2, constants);CHKERRQ(ierr);
    }

    Parameter *ctx;
    ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);


    // setup boundary
  PetscInt         id;
  id   = 3;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "top wall velocity",    "marker", 0, 0, NULL, (void (*)(void)) cubic_u, (void (*)(void)) cubic_u_t, 1, &id, ctx);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "bottom wall velocity", "marker", 0, 0, NULL, (void (*)(void)) cubic_u, (void (*)(void)) cubic_u_t, 1, &id, ctx);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "right wall velocity",  "marker", 0, 0, NULL, (void (*)(void)) cubic_u, (void (*)(void)) cubic_u_t, 1, &id, ctx);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(ds, DM_BC_ESSENTIAL, "left wall velocity",   "marker", 0, 0, NULL, (void (*)(void)) cubic_u, (void (*)(void)) cubic_u_t, 1, &id, ctx);CHKERRQ(ierr);


  ierr = PetscDSSetExactSolution(ds, 0, cubic_u, ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolutionTimeDerivative(ds, 0, cubic_u_t, ctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user) {

    PetscFunctionBeginUser;
    DM             cdm = dm;
    // Get the number of dimensions
    PetscInt dim;
    PetscErrorCode ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Number of Dim: %d", dim);

    PetscInt cStart;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);

    /* Create finite element */
    PetscFE fe; // PETSc object that manages a finite element space, e.g. the P_1 Lagrange element, this is the element!!!
    MPI_Comm        comm;
    ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);

    // determine if it a simplex element
    DMPolytopeType ct;
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
    ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
    PetscBool simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = PetscFECreateDefault(
            comm,//The comm
            dim,//the spatial dimension
            1, // the number of components, 1 for T
            simplex,
            "temp_",//The prefix for options
            PETSC_DEFAULT,//The quadrature order, 1 is default
            &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, "temperature");CHKERRQ(ierr);
    /* Set discretization and boundary conditions for each mesh */
    ierr = DMSetField(dm,
                      0,// The field number
                      NULL, // The label indicating the support of the field, or NULL for the entire mesh
                      (PetscObject) fe);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = DMCreateDS(dm);CHKERRQ(ierr);

    // Setup the real problem space
    ierr = SetupProblem(dm, user);CHKERRQ(ierr);
    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);//Calculate an index for the given PetscSection for the closure operation on the DM

//    while (cdm) {
////        ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);
//        ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
//        ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
//    }

    // remove one reference to FE
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

/* < v, n_e > */
static void objective(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[],  PetscInt numConstants, const PetscScalar constants[], PetscScalar *f0)
{
  const PetscReal rhoCp = PetscRealPart(constants[0]);
  f0[0] = u[0]*rhoCp;
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal time, Vec u, void *ctx)
{
    PetscFunctionBeginUser;


  DM             dm;
  MatNullSpace   nullsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  double    tt[1];


  PetscDS ds;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(ds, 0, objective);
  ierr = DMPlexComputeIntegralFEM(dm,u,tt,NULL);CHKERRQ(ierr);


   ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g, TotalEner = %f\n", (int) step, (double) time, (double) tt[0]);CHKERRQ(ierr);




    ierr = PetscObjectSetName((PetscObject) u, "Numerical Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

static PetscErrorCode initialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{

  PetscScalar r = PetscSqrtReal(PetscSqr(x[0] -.5) + PetscSqr(x[1] - .5));


  if (r < .25){
    u[0] = 100;
  }else{
    u[0] = 1;
  }

  return(0);
}



int demo(int argc,char **args)
{
    AppCtx          user; /* user-defined work context */
    PetscErrorCode ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
    // Check to see if there is a file for the mesh
    char meshFile[PETSC_MAX_PATH_LEN] = "";

    // Create a bag for the parameters, this allows them to be serialized
    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
    ierr = SetupParameters(&user);CHKERRQ(ierr);
    {
        Parameter     *p;
        char *name;
        ierr = PetscBagGetData(user.bag, (void **) &p);CHKERRQ(ierr);
        PetscBagGetName(user.bag, &name);
        PetscPrintf(PETSC_COMM_WORLD, "Input Parameters in %s:\n\t k: %f \n\t rhoCp: %f\n", name, p->k, p->rhoCp );
    }

    // check for options not in bag
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL,"Mesh Options","");CHKERRQ(ierr);
    ierr = PetscOptionsString("-f","Mesh","",meshFile,meshFile,sizeof(meshFile),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    // Create the time stepper
    TS ts;
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);

    // Create the dm, kinda of like a domain
    DM dm;
    CreateMesh(PETSC_COMM_WORLD, &user, &dm);

    // Sets the DM that may be used by some nonlinear solvers or preconditioners under the TS
    ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
    // Set a user context into a DM object, this results in it being passed into other calls
    ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

    // Setup the domain
    ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
    ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

    // get the vector
    Vec T;
    ierr = DMCreateGlobalVector(dm, &T);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) T, "Numerical Solution");CHKERRQ(ierr);

    // Tell the DMTS to use the FE functions
    ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &user);CHKERRQ(ierr);
    ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &user);CHKERRQ(ierr);
     ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &user);CHKERRQ(ierr);

    // Setup the time stepper
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
//    ierr = TSSetMaxTime(ts, 10.0);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    // update the time step for the current time
    PetscReal t;
    ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
    ierr = DMSetOutputSequenceNumber(dm, 0, t);CHKERRQ(ierr);


    // Set the initial
    PetscErrorCode (*func[1]) (PetscInt,PetscReal,const PetscReal [],PetscInt, PetscScalar *,void *);
    void            *ctxs[1];
    func[0] = cubic_u;
    ctxs[0] = &user;
    DMProjectFunctionLocal(dm, 0.0, func, ctxs, INSERT_ALL_VALUES, T);


//    VecAssemblyBegin(T);
//    VecAssemblyEnd(T);
    ierr = VecViewFromOptions(T, NULL, "-sol_start");CHKERRQ(ierr);


    ierr = TSMonitorSet(
            ts,
            MonitorError,
            &user,
            NULL
            );CHKERRQ(ierr);CHKERRQ(ierr);


    ierr = TSSolve(ts, T);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) T, "final_temp");CHKERRQ(ierr);
    ierr = VecViewFromOptions(T, NULL, "-sol_final");CHKERRQ(ierr);

    ierr = TSDestroy(&ts);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    /*
       Always call PetscFinalize() before exiting a program.  This routine
         - finalizes the PETSc libraries as well as MPI
         - provides summary and diagnostic information if certain runtime
           options are chosen (e.g., -log_view).
    */
    ierr = PetscFinalize();
    return ierr;
}

/*
Options:
    -dm_refine 1 :// sets the refine level on the mesh in the DM


 -dm_plex_separate_marker -snes_converged_reason  -dm_view vtk:sol.vtk  -sol_final vtk:sol.vtk::append -temp_petscspace_degree 1 -ts_monitor_solution draw -draw_pause 1 -snes_fd_color -snes_rtol 1E-10 -ts_monitor_solution_vtk filename-%03D.vtu -ts_fd_color

  ${PETSC_DIR}/arch-darwin-c-debug/bin/mpirun -n 4  ./Framework -dm_plex_separate_marker -snes_converged_reason  -sol_final vtk:sol.vtk::append -temp_petscspace_degree 1 -ts_monitor_solution draw -draw_pause 1 -snes_fd_color -snes_rtol 1E-10 -ts_monitor_solution_vtk filename-%03D.vtu -ts_fd_color -f "/Users/mcgurn/Desktop/test3.msh" -dm_view


 */
