//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cpd.cpp
//! \brief Initializes local spherical polar patch around a planet in a disk
//!
//! Physical parameters:
//! problem/qthermal: dimensionless thermal mass (default = 0)
//! problem/epsilon: gravitational softening length (default = 0)
//! problem/tinj: gradually introduce planet potential over this amount of time (default = 0)
//!
//! More optional parameters:
//! problem/axisymmetric: use azimuthally symmetric source terms even in 3D (default = false)
//! problem/rotation: enable centrifugal and coriolis source terms (default = true)
//! problem/stratified: enable vertical gravity and stratification (default = true)
//!
//! User boundary conditions
//! x1_inner: inner_hydro - [reflecting, outflow]
//!           inner_rad - [reflecting, outflow, outflow-isotropic]
//! x1_outer: outer_hydro - [fromfile, disk]
//!           outer_rad - [outflow-isotropic]
//! x2_inner: [reflecting]
//! x2_outer: [reflecting]

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <fstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../nr_radiation/radiation.hpp"

namespace {
Real qthermal, epsilon, tinj, kappa;
bool axisymmetric = false;
bool stratified = true;
bool rotation = true;
bool output_fluxes = false;
bool cylcoord = false;
Real sincosphi, sq_cosphi;
std::string outer_hydro;
std::string outer_rad;
std::string inner_hydro;
std::string inner_rad;
std::string outer_file;
// for outer boundary from file
int32_t fctype, fnvar;
int32_t fnx2, fnx1;
// for opacity from file
Real tempunit, rhounit, lunit;
std::string opacityfile;
static AthenaArray<Real> opacitytableross, opacitytableplanck;
static AthenaArray<Real> logttable;
static AthenaArray<Real> logrhotable;
} // namespace

void CopyIntensityPgen(Real *iri, Real *iro, int li, int lo, int n_ang);
void InterpolateBoundary(MeshBlock *pmb, int mode);
void InterpolateMeshBlock(MeshBlock *pmb, int mode);
void TableOpacity(MeshBlock *pmb, AthenaArray<Real> &prim);
bool ContainVariable(const std::string &haystack, const std::string &needle);
Real Potential(Real phi, Real theta, Real r, Real time);
// radiation boundary conditions

void ReflectInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void OutflowInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void OutflowIsoInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void OutflowIsoOuterX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReflectInnerX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void PolarWedgeInnerX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReflectOuterX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir, Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh);

// hydro boundary conditions
void ReflectInnerX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OutflowInnerX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void FromFileOuterX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReflectInnerX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void PolarWedgeInnerX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReflectOuterX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

// User-defined source term
void MySource(MeshBlock *pmb, const Real time, const Real dt,
                 const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
                 const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                 AthenaArray<Real> &cons_scalar);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  outer_hydro = pin->GetString("problem","outer_hydro");
  opacityfile = pin->GetOrAddString("problem","opacity_file","None");
  // data structures for outputting fluxes
  if (output_fluxes) {
    if (pmy_mesh->ndim == 1)
      AllocateUserOutputVariables(2);
    else if (pmy_mesh->ndim == 2)
      AllocateUserOutputVariables(4);
    else
      AllocateUserOutputVariables(6);
    SetUserOutputVariableName(0, "flux1in");
    SetUserOutputVariableName(1, "flux1out");
    if (pmy_mesh->f2) {
      SetUserOutputVariableName(2, "flux2in");
      SetUserOutputVariableName(3, "flux2out");
    }
    if (pmy_mesh->f3) {
      SetUserOutputVariableName(4, "flux3in");
      SetUserOutputVariableName(5, "flux3out");
    }
  }
  if (opacityfile != "None" && IM_RADIATION_ENABLED) {
    pnrrad->EnrollOpacityFunction(TableOpacity);
  }
  // if outer boundary fromfile
  if (outer_hydro == "fromfile") {
    AllocateRealUserMeshBlockDataField(1);
    ruser_meshblock_data[0].NewAthenaArray(NHYDRO,ncells2,NGHOST);
    InterpolateBoundary(this, 0);
  }
  return;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)
    cylcoord = true;
  qthermal = pin->GetOrAddReal("problem","qthermal",0.0);
  epsilon = pin->GetOrAddReal("problem","epsilon",0.0);
  axisymmetric = pin->GetOrAddBoolean("problem","axisymmetric",false);
  stratified = pin->GetOrAddBoolean("problem","stratified",true);
  rotation = pin->GetOrAddBoolean("problem","rotation",true);
  tinj = pin->GetOrAddReal("problem","tinj",0.0);
  kappa = pin->GetOrAddReal("problem","kappa",0.0);
  inner_hydro = pin->GetString("problem","inner_hydro");
  outer_hydro = pin->GetString("problem","outer_hydro");
  if (outer_hydro == "fromfile") {
    outer_file = pin->GetString("problem","outer_file");
  }
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    inner_rad = pin->GetString("problem","inner_rad");
    outer_rad = pin->GetString("problem","outer_rad");
  }
  tempunit = pin->GetOrAddReal("radiation","T_unit",0.0);
  rhounit = pin->GetOrAddReal("radiation","density_unit",0.0);
  lunit = pin->GetOrAddReal("radiation","length_unit",0.0);
  opacityfile = pin->GetOrAddString("problem","opacity_file","None");

  // parse input file output blocks to determine whether to output fluxes
  InputBlock *pib = pin->pfirst_block;
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 6, "output") == 0) {
      if (pin->DoesParameterExist(pib->block_name, "variable")) {
        std::string variable = pin->GetString(pib->block_name, "variable");
        if (ContainVariable(variable, "uov")) {
          output_fluxes = true;
        }
      }
    }
    pib = pib->pnext;
  }
  if ((ndim == 2) && (!cylcoord))
    axisymmetric = true;
  if (axisymmetric) {
    sq_cosphi = 0.5;
    sincosphi = 0.0;
  }
  // read opacity file
  if (opacityfile != "None" && IM_RADIATION_ENABLED) {
    opacitytableross.NewAthenaArray(70,140);
    opacitytableplanck.NewAthenaArray(70,140);
    logttable.NewAthenaArray(70);
    logrhotable.NewAthenaArray(140);
    FILE *fileopa;
    if ( (fileopa=fopen(opacityfile.c_str(),"r"))==NULL )
    {
      printf("Open input file error\n");
      return;
    }
    int i, j;
    Real rhoread,tread,rossread,planckread;
    char * line = NULL;
    size_t len = 0;
    getline(&line, &len, fileopa);
    for(j=0; j<140; j++){
      for(i=0; i<70; i++){
        fscanf(fileopa,"%lf %lf %lf %lf",&rhoread,&tread,&rossread,&planckread);
        logrhotable(j)=log10(rhoread);
        logttable(i)=log10(tread);
        opacitytableross(i,j)=rossread;
        opacitytableplanck(i,j)=planckread;
      }
    }
  }

  // enroll user-defined source term
  EnrollUserExplicitSourceFunction(MySource);
  // x1 inner boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    if (inner_hydro == "reflecting") {
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, ReflectInnerX1Hydro);
    }
    else if (inner_hydro == "outflow") {
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, OutflowInnerX1Hydro);
      std::cout << "outflow bc inner" << std::endl;
    }
    else {
      std::stringstream msg;
      msg << "### FATAL ERROR in ProblemGenerator" << std::endl
          << "problem/inner_hydro parameter in the input file" << std::endl
          << "must be one of [reflecting, outflow]" << std::endl;
      ATHENA_ERROR(msg);
    }
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      if (inner_rad == "reflecting") {
        EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, ReflectInnerX1Rad);
      }
      else if (inner_rad == "outflow") {
        EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, OutflowInnerX1Rad);
      }
      else if (inner_rad == "outflow-isotropic") {
        EnrollUserRadBoundaryFunction(BoundaryFace::inner_x1, OutflowIsoInnerX1Rad);
      }
      else {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator" << std::endl
            << "problem/inner_rad parameter in the input file" << std::endl
            << "must be one of [reflecting, outflow, outflow-isotropic]" << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  // x1 outer boundary condition
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    if (outer_hydro == "disk") {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1Hydro);
    }
    else if(outer_hydro == "fromfile") {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FromFileOuterX1Hydro);
    }
    else {
      std::stringstream msg;
      msg << "### FATAL ERROR in ProblemGenerator" << std::endl
          << "problem/outer_hydro parameter in the input file" << std::endl
          << "must be one of [disk, fromfile]" << std::endl;
      ATHENA_ERROR(msg);
    }
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      if (outer_rad == "outflow-isotropic") {
        EnrollUserRadBoundaryFunction(BoundaryFace::outer_x1, OutflowIsoOuterX1Rad);
      }
      else {
        std::stringstream msg;
        msg << "### FATAL ERROR in ProblemGenerator" << std::endl
            << "problem/outer_rad parameter in the input file" << std::endl
            << "must be one of [outflow-isotropic]" << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }

  // x2 inner boundary condition -- always reflecting
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, PolarWedgeInnerX2Hydro);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      EnrollUserRadBoundaryFunction(BoundaryFace::inner_x2, PolarWedgeInnerX2Rad);
    }
  }
  // x2 outer boundary condition -- always reflecting
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, ReflectOuterX2Hydro);
    if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
      EnrollUserRadBoundaryFunction(BoundaryFace::outer_x2, ReflectOuterX2Rad);
    }
  }

  // read file if outer boundary is "fromfile"
  if (outer_hydro == "fromfile") {
    std::ifstream file(outer_file, std::ios::binary);
    file.read ((char*)&fctype, sizeof(fctype));
    file.read ((char*)&fnvar, sizeof(fnvar));
    file.read ((char*)&fnx2, sizeof(fnx2));
    file.read ((char*)&fnx1, sizeof(fnx1));
    AllocateRealUserMeshDataField(3);
    ruser_mesh_data[0].NewAthenaArray(fnx2); // theta-grid
    ruser_mesh_data[1].NewAthenaArray(fnx1); // r-grid
    ruser_mesh_data[2].NewAthenaArray(NHYDRO,fnx2,fnx1); // data
    // read theta-grid
    for (int j=0; j<fnx2; j++) {
      double tmp;
      file.read ((char*)&tmp, sizeof(tmp));
      if (SINGLE_PRECISION_ENABLED)
        ruser_mesh_data[0](j) = float(tmp);
      else
        ruser_mesh_data[0](j) = tmp;
    }
    // read r-grid
    for (int i=0; i<fnx1; i++) {
      double tmp;
      file.read ((char*)&tmp, sizeof(tmp));
      if (SINGLE_PRECISION_ENABLED)
        ruser_mesh_data[1](i) = float(tmp);
      else
        ruser_mesh_data[1](i) = tmp;
    }
    // read data
    for (int n=0; n<fnvar; n++) {
      for (int j=0; j<fnx2; j++) {
        for (int i=0; i<fnx1; i++) {
          double tmp;
          file.read ((char*)&tmp, sizeof(tmp));
          if (SINGLE_PRECISION_ENABLED)
            ruser_mesh_data[2](n,j,i) = float(tmp);
          else
            ruser_mesh_data[2](n,j,i) = tmp;
        }
      }
    }
    // if file does not include pressure information, the set press = den
    if (fnvar < NHYDRO) {
      for (int j=0; j<fnx2; j++) {
        for (int i=0; i<fnx1; i++) {
          ruser_mesh_data[2](NHYDRO-1,j,i) = ruser_mesh_data[2](0,j,i);
        }
      }
    }
  }
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real x1, x2, x3, z, phi;
  Real gamma = peos->GetGamma();
  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        if (cylcoord) {
          phi = x2;
          z = x3;
        }
        else {
          phi = x3;
          z = x1*std::cos(x2);
        }
        if (!axisymmetric) {
          sq_cosphi = std::cos(phi)*std::cos(phi);
          sincosphi = std::sin(phi)*std::cos(phi);
        }
        if (stratified)
          phydro->u(IDN,k,j,i) = std::exp(-z*z/2.0);
        else
          phydro->u(IDN,k,j,i) = 1.0;
        if (rotation) {
          if (cylcoord) {
            phydro->u(IM1,k,j,i) = -1.5*x1*sincosphi;
            phydro->u(IM2,k,j,i) = -1.5*x1*sq_cosphi;
            phydro->u(IM3,k,j,i) = 0.0;
          }
          else {
            phydro->u(IM1,k,j,i) = -1.5*x1*std::sin(x2)*std::sin(x2)*sincosphi*phydro->u(IDN,k,j,i);
            phydro->u(IM2,k,j,i) = -1.5*x1*std::sin(x2)*std::cos(x2)*sincosphi*phydro->u(IDN,k,j,i);
            phydro->u(IM3,k,j,i) = -1.5*x1*std::sin(x2)*sq_cosphi*phydro->u(IDN,k,j,i);
          }
        }
        else {
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)/(gamma-1.0)
                               + SQR(phydro->u(IM1,k,j,i))/2.0/phydro->u(IDN,k,j,i)
                               + SQR(phydro->u(IM2,k,j,i))/2.0/phydro->u(IDN,k,j,i)
                               + SQR(phydro->u(IM3,k,j,i))/2.0/phydro->u(IDN,k,j,i);
        }
      }
    }
  }

  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    int nfreq = pnrrad->nfreq;
    AthenaArray<Real> ir_cm;
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
    Real *ir_lab;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          ir_lab = &(pnrrad->ir(k,j,i,0));
          for (int n=0; n<pnrrad->n_fre_ang; n++) {
             ir_lab[n] = 1.0;
          }
        }
      }
    }

    for (int k=0; k<ncells3; ++k) {
      for (int j=0; j<ncells2; ++j) {
        for (int i=0; i<ncells1; ++i) {
          for (int ifr=0; ifr < nfreq; ++ifr) {
            // sigma_a = sigma_p = sigma_pe reduces to short characteristics
            pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pnrrad->sigma_a(k,j,i,ifr) = phydro->u(IDN,k,j,i)*kappa;
            pnrrad->sigma_p(k,j,i,ifr) = phydro->u(IDN,k,j,i)*kappa;
            pnrrad->sigma_pe(k,j,i,ifr) = phydro->u(IDN,k,j,i)*kappa;
          }
        }
      }
    }
    ir_cm.DeleteAthenaArray();
  }
  return;
}

void TableOpacity(MeshBlock *pmb, AthenaArray<Real> &prim)
{
  Real gamma_gas = pmb->peos->GetGamma();
  NRRadiation *prad = pmb->pnrrad;
  int il = pmb->is; int jl = pmb->js; int kl = pmb->ks;
  int iu = pmb->ie; int ju = pmb->je; int ku = pmb->ke;
  il -= NGHOST;
  iu += NGHOST;
  if(ju > jl){
    jl -= NGHOST;
    ju += NGHOST;
  }
  if(ku > kl){
    kl -= NGHOST;
    ku += NGHOST;
  }

  for (int k=kl; k<=ku; ++k) {
  for (int j=jl; j<=ju; ++j) {
  for (int i=il; i<=iu; ++i) {
  for (int ifr=0; ifr<prad->nfreq; ++ifr){
    Real rho  = prim(IDN,k,j,i);
    Real gast = prim(IEN,k,j,i)/rho;
    Real logt = log10(gast*tempunit);
    Real logrho;
    int np1 = 0, np2 = 0, nt1=0, nt2=0;
    Real kappaA, kappaP;
    Real kappaS = 0.0;

    logrho = log10(rho*rhounit);

    // np1 < logrho < np2
    while((logrho > logrhotable(np2)) && (np2 < 140)){
        np1 = np2;
        np2++;
    }
    if(np2==140) np2=np1;

  // The data point should between NrhoT1 and NrhoT2
  // nt1 < logt < nt2
    while((logt > logttable(nt2)) && (nt2 < 70)){
        nt1 = nt2;
        nt2++;
    }
    if(nt2==70) nt2=nt1;

    Real kappaross_t1_p1=opacitytableross(nt1,np1);
    Real kappaross_t1_p2=opacitytableross(nt1,np2);
    Real kappaross_t2_p1=opacitytableross(nt2,np1);
    Real kappaross_t2_p2=opacitytableross(nt2,np2);
    Real kappaplanck_t1_p1=opacitytableplanck(nt1,np1);
    Real kappaplanck_t1_p2=opacitytableplanck(nt1,np2);
    Real kappaplanck_t2_p1=opacitytableplanck(nt2,np1);
    Real kappaplanck_t2_p2=opacitytableplanck(nt2,np2);

    Real p_1 = logrhotable(np1);
    Real p_2 = logrhotable(np2);
    Real t_1 = logttable(nt1);
    Real t_2 = logttable(nt2);
    if(np1 == np2){
      if(nt1 == nt2){
        kappaA = kappaross_t1_p1;
        kappaP = kappaplanck_t1_p1;
      }else{
        kappaA = kappaross_t1_p1 + (kappaross_t2_p1 - kappaross_t1_p1) *
                                (logt - t_1)/(t_2 - t_1);
        kappaP = kappaplanck_t1_p1 + (kappaplanck_t2_p1 - kappaplanck_t1_p1) *
                                (logt - t_1)/(t_2 - t_1);
      }// end same T
    }else{
      if(nt1 == nt2){
        kappaA = kappaross_t1_p1 + (kappaross_t1_p2 - kappaross_t1_p1) *
                                (logrho - p_1)/(p_2 - p_1);
        kappaP = kappaplanck_t1_p1 + (kappaplanck_t1_p2 - kappaplanck_t1_p1) *
                                (logrho - p_1)/(p_2 - p_1);
      }else{
        kappaA = kappaross_t1_p1 * (t_2 - logt) * (p_2 - logrho)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t2_p1 * (logt - t_1) * (p_2 - logrho)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t1_p2 * (t_2 - logt) * (logrho - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaross_t2_p2 * (logt - t_1) * (logrho - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1));
        kappaP = kappaplanck_t1_p1 * (t_2 - logt) * (p_2 - logrho)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t2_p1 * (logt - t_1) * (p_2 - logrho)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t1_p2 * (t_2 - logt) * (logrho - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1))
              + kappaplanck_t2_p2 * (logt - t_1) * (logrho - p_1)/
                                ((t_2 - t_1) * (p_2 - p_1));
      }
    }// end same p

    prad->sigma_s(k,j,i,ifr) = kappaS*rho*rhounit*lunit;
    prad->sigma_a(k,j,i,ifr) = kappaA*rho*rhounit*lunit;
    prad->sigma_p(k,j,i,ifr) = kappaP*rho*rhounit*lunit;
    prad->sigma_pe(k,j,i,ifr) = kappaP*rho*rhounit*lunit;
  }
  }}}
}

// hook for saving fluxes for outputting

void MeshBlock::UserWorkInStage(int stage) {
  if (output_fluxes) {
    Real coeff = 0.5;
    int nmax = 2*pmy_mesh->ndim;
    AthenaArray<Real> &x1flux = phydro->flux[X1DIR];
    AthenaArray<Real> &x2flux = phydro->flux[X2DIR];
    AthenaArray<Real> &x3flux = phydro->flux[X3DIR];
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (stage == 1) {
            for (int n=0; n<nmax; n++) {
              user_out_var(n,k,j,i) = 0.0;
            }
          }
          user_out_var(0,k,j,i) += coeff*x1flux(IDN,k,j,i);
          user_out_var(1,k,j,i) += coeff*x1flux(IDN,k,j,i+1);
          if (pmy_mesh->f2) {
            user_out_var(2,k,j,i) += coeff*x2flux(IDN,k,j,i);
            user_out_var(3,k,j,i) += coeff*x2flux(IDN,k,j+1,i);
          }
          if (pmy_mesh->f3) {
            user_out_var(4,k,j,i) += coeff*x3flux(IDN,k,j,i);
            user_out_var(5,k,j,i) += coeff*x3flux(IDN,k+1,j,i);
          }
        }
      }
    }
  }
  return;
}


void MeshBlock::UserWorkInLoop() {
  return;
}

void Mesh::UserWorkInLoop() {
  return;
}

Real Potential(Real x3, Real x2, Real x1, Real time) {
  Real pot, x, z;
  if (cylcoord) {
    pot = -qthermal/std::sqrt(x1*x1 + x3*x3 + epsilon*epsilon);
    x = x1*std::cos(x2);
    z = x3;
  }
  else {
    pot = -qthermal/std::sqrt(x1*x1 + epsilon*epsilon);
    x = x1*std::sin(x2)*std::cos(x3);
    z = x1*std::cos(x2);
  }
  if (time < tinj)
    pot *= SQR(std::sin(PI*time/tinj/2.0));
  if (rotation)
    pot += -1.5*x*x;
  if (stratified)
    pot += 0.5*z*z;
  return pot;
}

void MySource(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  AthenaArray<Real> x1area;
  AthenaArray<Real> vol;
  x1area.NewAthenaArray(pmb->ncells1+1);
  vol.NewAthenaArray(pmb->ncells1);
  AthenaArray<Real> x2aream;
  AthenaArray<Real> x2areap;
  x2aream.NewAthenaArray(pmb->ncells1);
  x2areap.NewAthenaArray(pmb->ncells1);
  AthenaArray<Real> x3aream;
  AthenaArray<Real> x3areap;
  x3aream.NewAthenaArray(pmb->ncells1);
  x3areap.NewAthenaArray(pmb->ncells1);
  Real phi, z;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    Real x3m = pmb->pcoord->x3f(k);
    Real x3p = pmb->pcoord->x3f(k+1);
    for (int j = pmb->js; j <= pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      Real x2m = pmb->pcoord->x2f(j);
      Real x2p = pmb->pcoord->x2f(j+1);
      pmb->pcoord->Face1Area(k, j, pmb->is, pmb->ie+1, x1area);
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
      if (pmb->block_size.nx2 > 1) {
        pmb->pcoord->Face2Area(k, j,   pmb->is, pmb->ie, x2aream);
        pmb->pcoord->Face2Area(k, j+1, pmb->is, pmb->ie, x2areap);
      }
      if (pmb->block_size.nx3 > 1) {
        pmb->pcoord->Face3Area(k,   j, pmb->is, pmb->ie, x3aream);
        pmb->pcoord->Face3Area(k+1, j, pmb->is, pmb->ie, x3areap);
      }
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real x1m = pmb->pcoord->x1f(i);
        Real x1p = pmb->pcoord->x1f(i+1);
        // conservative energy source terms
        if (NON_BAROTROPIC_EOS) {
          Real phil = Potential(x3, x2, x1m, time);
          Real phic = Potential(x3, x2, x1,  time);
          Real phir = Potential(x3, x2, x1p, time);
          cons(IEN,k,j,i) -= dt*(pmb->phydro->flux[X1DIR](IDN,k,j,i+1)*x1area(i+1)*(phir - phic)
                                +pmb->phydro->flux[X1DIR](IDN,k,j,i)*x1area(i)*(phic - phil))/vol(i);
          if (pmb->block_size.nx2 > 1) {
            phil = Potential(x3, x2m, x1, time);
            phic = Potential(x3, x2,  x1,  time);
            phir = Potential(x3, x2p, x1, time);
            cons(IEN,k,j,i) -= dt*(pmb->phydro->flux[X2DIR](IDN,k,j+1,i)*x2areap(i)*(phir - phic)
                                  +pmb->phydro->flux[X2DIR](IDN,k,j,i)*x2aream(i)*(phic - phil))/vol(i);
          }
          if (pmb->block_size.nx3 > 1) {
            phil = Potential(x3m, x2, x1, time);
            phic = Potential(x3,  x2, x1,  time);
            phir = Potential(x3p, x2, x1, time);
            cons(IEN,k,j,i) -= dt*(pmb->phydro->flux[X3DIR](IDN,k+1,j,i)*x3areap(i)*(phir - phic)
                                  +pmb->phydro->flux[X3DIR](IDN,k,j,i)*x3aream(i)*(phic - phil))/vol(i);
          }
        }
        // if non-axisymmetric expand phi terms
        if (cylcoord) {
          phi = x2;
          z = x3;
        }
        else {
          phi = x3;
          z = x1*std::cos(x2);
        }
        if (!axisymmetric){
            sq_cosphi = std::cos(phi)*std::cos(phi);
            sincosphi = std::sin(phi)*std::cos(phi);
        }

        // planet gravity
        Real rsq = x1*x1;
        if (cylcoord)
          rsq += x3*x3;
        Real g = -qthermal/std::pow(rsq + epsilon*epsilon, 1.5);
        if (time < tinj)
          g *= SQR(std::sin(PI*time/tinj/2.0));
        cons(IM1,k,j,i) += x1*prim(IDN,k,j,i)*g*dt;
        if (cylcoord)
          cons(IM3,k,j,i) += x3*prim(IDN,k,j,i)*g*dt;

        if (rotation) {
          // coriolis force
          if (cylcoord) {
            cons(IM1,k,j,i) += 2.0*prim(IDN,k,j,i)*prim(IM2,k,j,i)*dt;
            cons(IM2,k,j,i) -= 2.0*prim(IDN,k,j,i)*prim(IM1,k,j,i)*dt;
          }
          else {
            cons(IM1,k,j,i) += 2.0*std::sin(x2)*prim(IDN,k,j,i)*prim(IM3,k,j,i)*dt;
            cons(IM2,k,j,i) += 2.0*std::cos(x2)*prim(IDN,k,j,i)*prim(IM3,k,j,i)*dt;
            cons(IM3,k,j,i) -= 2.0*std::sin(x2)*prim(IDN,k,j,i)*prim(IM1,k,j,i)*dt;
            cons(IM3,k,j,i) -= 2.0*std::cos(x2)*prim(IDN,k,j,i)*prim(IM2,k,j,i)*dt;
          }

          // centrifugal shearing force
          if (cylcoord) {
            cons(IM1,k,j,i) += 3.0*prim(IDN,k,j,i)*x1*sq_cosphi*dt;
            cons(IM2,k,j,i) -= 3.0*prim(IDN,k,j,i)*x1*sincosphi*dt;
          }
          else {
            cons(IM1,k,j,i) += 3.0*prim(IDN,k,j,i)*x1*std::sin(x2)*std::sin(x2)*sq_cosphi*dt;
            cons(IM2,k,j,i) += 3.0*prim(IDN,k,j,i)*x1*std::sin(x2)*std::cos(x2)*sq_cosphi*dt;
            cons(IM3,k,j,i) -= 3.0*prim(IDN,k,j,i)*x1*std::sin(x2)*sincosphi*dt;
          }
        }

        // centrifugal vertical force
        if (stratified) {
          if (cylcoord)
            cons(IM3,k,j,i) -= prim(IDN,k,j,i)*z;
          else {
            cons(IM1,k,j,i) -= prim(IDN,k,j,i)*z*std::cos(x2)*dt;
            cons(IM2,k,j,i) += prim(IDN,k,j,i)*z*std::sin(x2)*dt;
          }
        }
      }
    }
  }
  vol.DeleteAthenaArray();
  x1area.DeleteAthenaArray();
  x2aream.DeleteAthenaArray();
  x2areap.DeleteAthenaArray();
  x3aream.DeleteAthenaArray();
  x3areap.DeleteAthenaArray();
  return;
}

void CopyIntensityPgen(Real *iri, Real *iro, int li, int lo, int n_ang) {
  // here ir is only intensity for each cell and each frequency band
  for (int n=0; n<n_ang; ++n) {
    int angi = li * n_ang + n;
    int ango = lo * n_ang + n;
    iro[angi] = iri[ango];
    iro[ango] = iri[angi];
  }
}

void ReflectInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy radiation variables into ghost zones,
  // reflect rays along angles with opposite nx
  const int& noct = pmb->pnrrad->noct;
  int n_ang = pmb->pnrrad->nang/noct; // angles per octant
  const int& nfreq = pmb->pnrrad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          Real *iri = &ir(k,j,(is+i-1),ifr*pmb->pnrrad->nang);
          Real *iro = &ir(k,j, is-i, ifr*pmb->pnrrad->nang);
          CopyIntensityPgen(iri, iro, 0, 1, n_ang);
          if (noct > 2) {
            CopyIntensityPgen(iri, iro, 2, 3, n_ang);
          }
          if (noct > 3) {
            CopyIntensityPgen(iri, iro, 4, 5, n_ang);
            CopyIntensityPgen(iri, iro, 6, 7, n_ang);
          }
        }
      }
    }
  }
  return;
}

void OutflowInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy radiation variables into ghost zones,
  const int& nang = pmb->pnrrad->nang; // angles per octant
  const int& nfreq = pmb->pnrrad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for (int n=0; n<nang; ++n) {
            int ang=ifr*nang+n;
            ir(k,j,is-i,ang) = ir(k,j,is,ang);
          }
        }
      }
    }
  }
  return;
}

void OutflowIsoInnerX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy radiation variables into ghost zones,
  const int& nang = pmb->pnrrad->nang; // angles per octant
  const int& nfreq = pmb->pnrrad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for (int n=0; n<nang; ++n) {
            int ang=ifr*nang+n;
            Real miux = prad->mu(0,k,j,ie,n);
            if (miux < 0.0) {
              ir(k,j,is-i,ang) = ir(k,j,is,ang);
            }
            else {
              Real temp = pmb->phydro->w(IPR,k,j,is-i)/pmb->phydro->w(IDN,k,j,is-i);
              ir(k,j,is-i,ang) = temp*temp*temp*temp;
            }
          }
        }
      }
    }
  }
  return;
}

void OutflowIsoOuterX1Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  int nang = prad->nang; // angles per octant
  int nfreq = prad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n) {
            int ang=ifr*nang+n;
            Real miux = prad->mu(0,k,j,ie,n);
            if (miux > 0.0) {
              ir(k,j,ie+i,ang) = ir(k,j,ie,ang);
            } else {
              Real temp = pmb->phydro->w(IPR,k,j,ie+i)/pmb->phydro->w(IDN,k,j,ie+i);
              ir(k,j,ie+i,ang) = temp*temp*temp*temp;
            }
          }
	}
      }
    }
  }
  return;
}

void ReflectInnerX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy radiation variables into ghost zones,
  // reflect rays along angles with opposite nx
  const int& noct = pmb->pnrrad->noct;
  int n_ang = pmb->pnrrad->nang/noct; // angles per octant
  const int& nfreq = pmb->pnrrad->nfreq; // number of frequency bands

  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          Real *iri = &ir(k,js+j-1,i,ifr*pmb->pnrrad->nang);
          Real *iro = &ir(k,js-j,i, ifr*pmb->pnrrad->nang);
          CopyIntensityPgen(iri, iro, 0, 2, n_ang);
          CopyIntensityPgen(iri, iro, 1, 3, n_ang);

          if (noct > 3) {
            CopyIntensityPgen(iri, iro, 4, 6, n_ang);
            CopyIntensityPgen(iri, iro, 5, 7, n_ang);
          }
        }
      }
    }
  }
  return;
}

void PolarWedgeInnerX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // do not flip the sign of specific intensity
  int nang = prad->nang; // angles per octant
  int nfreq = prad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          for(int n=0; n<nang; ++n) {
            int ang=ifr*nang+n;
            ir(k,js-j,i,ang) = ir(k,js+j-1,i,ang);
          }
	}
      }
    }
  }
  return;
}

void ReflectOuterX2Rad(MeshBlock *pmb, Coordinates *pco, NRRadiation *prad,
              const AthenaArray<Real> &w, FaceField &b,
              AthenaArray<Real> &ir,
              Real time, Real dt,
              int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy radiation variables into ghost zones,
  // reflect rays along angles with opposite nx

  const int& noct = pmb->pnrrad->noct;
  int n_ang = pmb->pnrrad->nang/noct; // angles per octant
  const int& nfreq = pmb->pnrrad->nfreq; // number of frequency bands
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        for (int ifr=0; ifr<nfreq; ++ifr) {
          Real *iri = &ir(k,je-j+1,i,ifr*pmb->pnrrad->nang);
          Real *iro = &ir(k,je+j,i, ifr*pmb->pnrrad->nang);
          CopyIntensityPgen(iri, iro, 0, 2, n_ang);
          CopyIntensityPgen(iri, iro, 1, 3, n_ang);

          if (noct > 3) {
            CopyIntensityPgen(iri, iro, 4, 6, n_ang);
            CopyIntensityPgen(iri, iro, 5, 7, n_ang);
          }
        }
      }
    }
  }
  return;
}

void ReflectInnerX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          if (n == IVX) {
            prim(n,k,j,is-i) = -prim(n,k,j,is+i-1);
          }
          else {
            prim(n,k,j,is-i) = prim(n,k,j,is+i-1);
          }
        }
      }
    }
  }
}

void OutflowInnerX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is);
	}
      }
    }
  }
}

void DiskOuterX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  Real z, phi;
  for (int k=ks; k<=ke; ++k) {
    Real x3 = pco->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        Real x1 = pco->x1v(ie+i);
        if (cylcoord) {
          phi = x2;
          z = x3;
        }
        else {
          phi = x3;
          z = x1*std::cos(x2);
        }
        if (!axisymmetric) {
          sq_cosphi = std::cos(phi)*std::cos(phi);
          sincosphi = std::sin(phi)*std::cos(phi);
        }
	if (stratified)
          prim(IDN,k,j,ie+i) = std::exp(-z*z/2.0);
        else
          prim(IDN,k,j,ie+i) = 1.0;
        if (rotation) {
          if (cylcoord) {
            prim(IVX,k,j,ie+i) = -1.5*x1*sincosphi;
            prim(IVY,k,j,ie+i) = -1.5*x1*sq_cosphi;
            prim(IVZ,k,j,ie+i) = 0.0;
          }
          else {
            prim(IVX,k,j,ie+i) = -1.5*x1*std::sin(x2)*std::sin(x2)*sincosphi;
            prim(IVY,k,j,ie+i) = -1.5*x1*std::sin(x2)*std::cos(x2)*sincosphi;
            prim(IVZ,k,j,ie+i) = -1.5*x1*std::sin(x2)*sq_cosphi;
          }
        }
        else {
          prim(IVX,k,j,ie+i) = 0.0;
          prim(IVY,k,j,ie+i) = 0.0;
          prim(IVZ,k,j,ie+i) = 0.0;
        }
        if (NON_BAROTROPIC_EOS) {
          prim(IPR,k,j,ie+i) = prim(IDN,k,j,ie+i);
        }
      }
    }
  }
}

void FromFileOuterX1Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int j=js; j<=je; ++j) {
    for (int i=1; i<=ngh; ++i) {
      prim(IDN,0,j,ie+i) = pmb->ruser_meshblock_data[0](0,j,i-1);
      prim(IVX,0,j,ie+i) = pmb->ruser_meshblock_data[0](1,j,i-1);
      prim(IVY,0,j,ie+i) = pmb->ruser_meshblock_data[0](2,j,i-1);
      prim(IVZ,0,j,ie+i) = pmb->ruser_meshblock_data[0](3,j,i-1);
      if (NON_BAROTROPIC_EOS) {
        prim(IPR,0,j,ie+i) = pmb->ruser_meshblock_data[0](4,j,i-1);
      }
    }
  }
}

void ReflectInnerX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          if (n == IVY) {
            prim(n,k,js-j,i) = -prim(n,k,js+j-1,i);
          }
          else {
            prim(n,k,js-j,i) = prim(n,k,js+j-1,i);
          }
        }
      }
    }
  }
}

void PolarWedgeInnerX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  const bool pole_flip[] = {false, false, true, true, false};
  for (int n=0; n<NHYDRO; ++n) {
    Real sign = pole_flip[n] ? -1.0 : 1.0;
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
            prim(n,k,js-j,i) = sign * prim(n,k,js+j-1,i);
        }
      }
    }
  }
}

void ReflectOuterX2Hydro(
    MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh) {
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
	for (int i=is; i<=ie; ++i) {
          if (n == IVY) {
            prim(n,k,je+j,i) = -prim(n,k,je-j+1,i);
          }
          else {
            prim(n,k,je+j,i) = prim(n,k,je-j+1,i);
          }
        }
      }
    }
  }
}

// mode {0:nearest, 1:linear}
void InterpolateBoundary(MeshBlock *pmb, int mode) {
  for (int n=0; n<NHYDRO; n++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real theta = pmb->pcoord->x2v(j);
      // calculate j location on interpolation grid
      int j_index = 0;
      if (theta <= pmb->pmy_mesh->ruser_mesh_data[0](0)) {
        j_index = 0;
      }
      else if (theta >= pmb->pmy_mesh->ruser_mesh_data[0](fnx2-1)) {
        j_index = fnx2-1;
      }
      else {
        while (theta > pmb->pmy_mesh->ruser_mesh_data[0](j_index)) {
          j_index += 1;
        }
      }
      for (int i=0; i<NGHOST; i++) {
        Real r = pmb->pcoord->x1v(pmb->ie+i+1);
        // calculate i location on interpolation grid
        int i_index = 0;
        if (r <= pmb->pmy_mesh->ruser_mesh_data[1](0)) {
          i_index = 0;
        }
        else if (r >= pmb->pmy_mesh->ruser_mesh_data[1](fnx1-1)) {
          i_index = fnx1-1;
        }
        else {
          while (r > pmb->pmy_mesh->ruser_mesh_data[1](i_index)) {
            i_index += 1;
          }
        }
        if (mode == 0) {
          pmb->ruser_meshblock_data[0](n,j,i) = pmb->pmy_mesh->ruser_mesh_data[2](n,j_index,i_index);
        }
      }
    }
  }
  return;
}

// mode {0:nearest, 1:linear}
void InterpolateMeshBlock(MeshBlock *pmb, int mode) {
  for (int n=0; n<NHYDRO; n++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      Real theta = pmb->pcoord->x2v(j);
      // calculate j location on interpolation grid
      int j_index = 0;
      if (theta <= pmb->pmy_mesh->ruser_mesh_data[0](0)) {
        j_index = 0;
      }
      else if (theta >= pmb->pmy_mesh->ruser_mesh_data[0](fnx2-1)) {
        j_index = fnx2-1;
      }
      else {
        while (theta > pmb->pmy_mesh->ruser_mesh_data[0](j_index)) {
          j_index += 1;
        }
      }
      for (int i=pmb->is; i<=pmb->ie; i++) {
        Real r = pmb->pcoord->x1v(i);
        // calculate i location on interpolation grid
        int i_index = 0;
        if (r <= pmb->pmy_mesh->ruser_mesh_data[1](0)) {
          i_index = 0;
        }
        else if (r >= pmb->pmy_mesh->ruser_mesh_data[1](fnx1-1)) {
          i_index = fnx1-1;
        }
        else {
          while (r > pmb->pmy_mesh->ruser_mesh_data[1](i_index)) {
            i_index += 1;
          }
        }
        if (mode == 0) {
          pmb->ruser_meshblock_data[0](n,j,i) = pmb->pmy_mesh->ruser_mesh_data[2](n,j_index,i_index);
        }
      }
    }
  }
  return;
}

bool ContainVariable(const std::string &haystack, const std::string &needle) {
  if (haystack.compare(needle) == 0)
    return true;
  if (haystack.find(',' + needle + ',') != std::string::npos)
    return true;
  if (haystack.find(needle + ',') == 0)
    return true;
  if (haystack.find(',' + needle) != std::string::npos
      && haystack.find(',' + needle) == haystack.length() - needle.length() - 1)
    return true;
  return false;
}
