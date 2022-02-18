/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/qcd/action/fermion/WilsonExpCloverFermionImplementation.h

    Copyright (C) 2017 - 2022

    Author: paboyle <paboyle@ph.ed.ac.uk>
    Author: Guido Cossu <guido.cossu@ed.ac.uk>
    Author: Daniel Richtmann <daniel.richtmann@gmail.com>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
/*  END LEGAL */

#include <Grid/Grid.h>
#include <Grid/qcd/spin/Dirac.h>
#include <Grid/qcd/action/fermion/WilsonExpCloverFermion.h>

NAMESPACE_BEGIN(Grid);

template<class Impl>
WilsonExpCloverFermion<Impl>::WilsonExpCloverFermion(GaugeField&                         _Umu,
                                               GridCartesian&                      Fgrid,
                                               GridRedBlackCartesian&              Hgrid,
                                               const RealD                         _mass,
                                               const RealD                         _twmass,
                                               const RealD                         _csw_r,
                                               const RealD                         _csw_t,
                                               const WilsonAnisotropyCoefficients& clover_anisotropy,
                                               const ImplParams&                   impl_p)
  : WilsonFermion<Impl>(_Umu, Fgrid, Hgrid, _mass, impl_p, clover_anisotropy)
  , CloverTerm(&Fgrid)
  , ExpCloverTerm(&Fgrid)
  , ExpCloverTermInv(&Fgrid)
  , ExpCloverTermEven(&Hgrid)
  , ExpCloverTermOdd(&Hgrid)
  , ExpCloverTermInvEven(&Hgrid)
  , ExpCloverTermInvOdd(&Hgrid)
  , ExpCloverTermDagEven(&Hgrid)
  , ExpCloverTermDagOdd(&Hgrid)
  , ExpCloverTermInvDagEven(&Hgrid)
  , ExpCloverTermInvDagOdd(&Hgrid) {
  assert(Nd == 4); // require 4 dimensions

  if(clover_anisotropy.isAnisotropic) {
    csw_r     = _csw_r * 0.5 / clover_anisotropy.xi_0;
    diag_mass = _mass + 1.0 + (Nd - 1) * (clover_anisotropy.nu / clover_anisotropy.xi_0);
  } else {
    csw_r     = _csw_r * 0.5;
    diag_mass = 4.0 + _mass;
  }
  csw_t = _csw_t * 0.5;

  if(csw_r == 0)
    std::cout << GridLogWarning << "Initializing WilsonCloverFermion with csw_r = 0" << std::endl;
  if(csw_t == 0)
    std::cout << GridLogWarning << "Initializing WilsonCloverFermion with csw_t = 0" << std::endl;

  twmass = _twmass;

  ImportGauge(_Umu);
}

// *NOT* EO
template <class Impl>
void WilsonExpCloverFermion<Impl>::M(const FermionField &in, FermionField &out)
{
  FermionField temp(out.Grid());

  // Wilson term
  out.Checkerboard() = in.Checkerboard();
  this->Dhop(in, out, DaggerNo);

  // Clover term
  Mooee(in, temp);

  out += temp;
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::Mdag(const FermionField &in, FermionField &out)
{
  FermionField temp(out.Grid());

  // Wilson term
  out.Checkerboard() = in.Checkerboard();
  this->Dhop(in, out, DaggerYes);

  // Clover term
  MooeeDag(in, temp);

  out += temp;
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::ImportGauge(const GaugeField &_Umu)
{
  double t0 = usecond();
  WilsonFermion<Impl>::ImportGauge(_Umu);
  double t1 = usecond();
  GridBase *grid = _Umu.Grid();
  typename Impl::GaugeLinkField Bx(grid), By(grid), Bz(grid), Ex(grid), Ey(grid), Ez(grid);

  double t2 = usecond();
  // Compute the field strength terms mu>nu
  WilsonLoops<Impl>::FieldStrength(Bx, _Umu, Zdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(By, _Umu, Zdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Bz, _Umu, Ydir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ex, _Umu, Tdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ey, _Umu, Tdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(Ez, _Umu, Tdir, Zdir);

  double t3 = usecond();
  // Compute the Clover Operator acting on Colour and Spin
  // multiply here by the clover coefficients for the anisotropy
  CloverTerm  = Helpers::fillCloverYZ(Bx) * csw_r;
  CloverTerm += Helpers::fillCloverXZ(By) * csw_r;
  CloverTerm += Helpers::fillCloverXY(Bz) * csw_r;
  CloverTerm += Helpers::fillCloverXT(Ex) * csw_t;
  CloverTerm += Helpers::fillCloverYT(Ey) * csw_t;
  CloverTerm += Helpers::fillCloverZT(Ez) * csw_t;

  double t4 = usecond();

  int lvol = _Umu.Grid()->lSites();
  int DimRep = Impl::Dimension;

  // Exponentiate

  {

   typedef iMatrix<ComplexD,6> mat;

   autoView(CTv,CloverTerm,CpuRead);
   autoView(CTExpv,ExpCloverTerm,CpuWrite);

   thread_for(site, lvol, {
    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    mat srcCloverOpUL(0.0); // upper left block
    mat srcCloverOpLR(0.0); // lower right block
    mat ExpCloverOp;

    typename SiteClover::scalar_object Qx = Zero(), Qxexp = Zero();

    peekLocalSite(Qx, CTv, lcoor);

    // exp(A)

    //
    // upper left block
    //

    for (int j = 0; j < Ns/2; j++)
     for (int k = 0; k < Ns/2; k++)
      for (int a = 0; a < DimRep; a++)
       for (int b = 0; b < DimRep; b++){
        auto zz =  Qx()(j, k)(a, b);
        srcCloverOpUL(a + j * DimRep, b + k * DimRep) = std::complex<double>(zz);
       }

    ExpCloverOp = Exponentiate(srcCloverOpUL,1.0/(diag_mass));

    for (int j = 0; j < Ns/2; j++)
     for (int k = 0; k < Ns/2; k++)
      for (int a = 0; a < DimRep; a++)
       for (int b = 0; b < DimRep; b++)
        Qxexp()(j, k)(a, b) = ExpCloverOp(a + j * DimRep, b + k * DimRep);

    //
    // lower right block
    //

    for (int j = 0; j < Ns/2; j++)
     for (int k = 0; k < Ns/2; k++)
      for (int a = 0; a < DimRep; a++)
       for (int b = 0; b < DimRep; b++){
        auto zz =  Qx()(j+Ns/2, k+Ns/2)(a, b);
        srcCloverOpLR(a + j * DimRep, b + k * DimRep) = std::complex<double>(zz);
       }


    ExpCloverOp = Exponentiate(srcCloverOpLR,1.0/(diag_mass));

    for (int j = 0; j < Ns/2; j++)
     for (int k = 0; k < Ns/2; k++)
      for (int a = 0; a < DimRep; a++)
       for (int b = 0; b < DimRep; b++)
        Qxexp()(j+Ns/2, k+Ns/2)(a, b) = ExpCloverOp(a + j * DimRep, b + k * DimRep);

    // Now that the full 12x12 block is filled do poke!
    pokeLocalSite(Qxexp, CTExpv, lcoor);
   });
  }
  ExpCloverTerm *= diag_mass;

  // Add twisted mass
  CloverField T(CloverTerm.Grid());
  T = Zero();
  autoView(T_v,T,CpuWrite);
  thread_for(i, CloverTerm.Grid()->oSites(),
  {
    T_v[i]()(0, 0) = +twmass;
    T_v[i]()(1, 1) = +twmass;
    T_v[i]()(2, 2) = -twmass;
    T_v[i]()(3, 3) = -twmass;
  });
  T = timesI(T);
  ExpCloverTerm += T;

  double t5 = usecond();
  {
    autoView(CTExpv,ExpCloverTerm,CpuRead);
    autoView(CTExpIv,ExpCloverTermInv,CpuWrite);
    thread_for(site, lvol, {
      Coordinate lcoor;
      grid->LocalIndexToLocalCoor(site, lcoor);
      Eigen::MatrixXcd EigenCloverOp = Eigen::MatrixXcd::Zero(Ns/2 * DimRep, Ns/2 * DimRep);
      Eigen::MatrixXcd EigenInvCloverOp = Eigen::MatrixXcd::Zero(Ns/2 * DimRep, Ns/2 * DimRep);
      typename SiteClover::scalar_object Qx = Zero(), Qxinv = Zero();
      peekLocalSite(Qx, CTExpv, lcoor);

      //
      // upper left block
      //

      for (int j = 0; j < Ns/2; j++)
       for (int k = 0; k < Ns/2; k++)
        for (int a = 0; a < DimRep; a++)
         for (int b = 0; b < DimRep; b++){
          auto zz =  Qx()(j, k)(a, b);
          EigenCloverOp(a + j * DimRep, b + k * DimRep) = std::complex<double>(zz);
         }

      EigenInvCloverOp = EigenCloverOp.inverse();

      for (int j = 0; j < Ns/2; j++)
       for (int k = 0; k < Ns/2; k++)
        for (int a = 0; a < DimRep; a++)
         for (int b = 0; b < DimRep; b++)
          Qxinv()(j, k)(a, b) = EigenInvCloverOp(a + j * DimRep, b + k * DimRep);

      //
      // lower right block
      //

      for (int j = 0; j < Ns/2; j++)
       for (int k = 0; k < Ns/2; k++)
        for (int a = 0; a < DimRep; a++)
         for (int b = 0; b < DimRep; b++){
          auto zz =  Qx()(j+Ns/2, k+Ns/2)(a, b);
          EigenCloverOp(a + j * DimRep, b + k * DimRep) = std::complex<double>(zz);
         }

      EigenInvCloverOp = EigenCloverOp.inverse();

      for (int j = 0; j < Ns/2; j++)
       for (int k = 0; k < Ns/2; k++)
        for (int a = 0; a < DimRep; a++)
         for (int b = 0; b < DimRep; b++)
          Qxinv()(j+Ns/2, k+Ns/2)(a, b) = EigenInvCloverOp(a + j * DimRep, b + k * DimRep);

      // Now that the full 12x12 block is filled do poke!
      pokeLocalSite(Qxinv, CTExpIv, lcoor);
    });
  }

  double t6 = usecond();
  // Separate the even and odd parts
  pickCheckerboard(Even, ExpCloverTermEven, ExpCloverTerm);
  pickCheckerboard(Odd, ExpCloverTermOdd, ExpCloverTerm);

  pickCheckerboard(Even, ExpCloverTermDagEven, adj(ExpCloverTerm));
  pickCheckerboard(Odd, ExpCloverTermDagOdd, adj(ExpCloverTerm));

  pickCheckerboard(Even, ExpCloverTermInvEven, ExpCloverTermInv);
  pickCheckerboard(Odd, ExpCloverTermInvOdd, ExpCloverTermInv);

  pickCheckerboard(Even, ExpCloverTermInvDagEven, adj(ExpCloverTermInv));
  pickCheckerboard(Odd, ExpCloverTermInvDagOdd, adj(ExpCloverTermInv));
  double t7 = usecond();

#if 0
  std::cout << GridLogMessage << "WilsonCloverFermion::ImportGauge timings:"
            << " WilsonFermion::Importgauge = " << (t1 - t0) / 1e6
            << ", allocations = "               << (t2 - t1) / 1e6
            << ", field strength = "            << (t3 - t2) / 1e6
            << ", fill clover = "               << (t4 - t3) / 1e6
            << ", exponentiation + twmass = "   << (t5 - t4) / 1e6
            << ", inversions = "                << (t6 - t5) / 1e6
            << ", pick cbs = "                  << (t7 - t6) / 1e6
            << ", total = "                     << (t7 - t0) / 1e6
            << std::endl;
#endif
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::Mooee(const FermionField &in, FermionField &out)
{
  this->MooeeInternal(in, out, DaggerNo, InverseNo);
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::MooeeDag(const FermionField &in, FermionField &out)
{
  this->MooeeInternal(in, out, DaggerYes, InverseNo);
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::MooeeInv(const FermionField &in, FermionField &out)
{
  this->MooeeInternal(in, out, DaggerNo, InverseYes);
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::MooeeInvDag(const FermionField &in, FermionField &out)
{
  this->MooeeInternal(in, out, DaggerYes, InverseYes);
}

template <class Impl>
void WilsonExpCloverFermion<Impl>::MooeeInternal(const FermionField &in, FermionField &out, int dag, int inv)
{
  out.Checkerboard() = in.Checkerboard();
  CloverField *eClover;
  assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);

  if (dag)
  {
    if (in.Grid()->_isCheckerBoarded)
    {
      if (in.Checkerboard() == Odd)
      {
        eClover = (inv) ? &ExpCloverTermInvDagOdd : &ExpCloverTermDagOdd;
      }
      else
      {
        eClover = (inv) ? &ExpCloverTermInvDagEven : &ExpCloverTermDagEven;
      }
      Helpers::multCloverField(out, *eClover, in);
    }
    else
    {
      eClover = (inv) ? &ExpCloverTermInv : &ExpCloverTerm;
      Helpers::multCloverField(out, *eClover, in); // don't bother with adj, hermitian anyway
    }
  }
  else
  {
    if (in.Grid()->_isCheckerBoarded)
    {

      if (in.Checkerboard() == Odd)
      {
        //  std::cout << "Calling clover term Odd" << std::endl;
        eClover = (inv) ? &ExpCloverTermInvOdd : &ExpCloverTermOdd;
      }
      else
      {
        //  std::cout << "Calling clover term Even" << std::endl;
        eClover = (inv) ? &ExpCloverTermInvEven : &ExpCloverTermEven;
      }
      Helpers::multCloverField(out, *eClover, in);
      //  std::cout << GridLogMessage << "*Clover.Checkerboard() "  << (*Clover).Checkerboard() << std::endl;
    }
    else
    {
      eClover = (inv) ? &ExpCloverTermInv : &ExpCloverTerm;
      Helpers::multCloverField(out, *eClover, in);
    }
  }
} // MooeeInternal

// Derivative parts unpreconditioned pseudofermions
template <class Impl>
void WilsonExpCloverFermion<Impl>::MDeriv(GaugeField &force, const FermionField &X, const FermionField &Y, int dag)
{
  assert(0);
}

// Derivative parts
template <class Impl>
void WilsonExpCloverFermion<Impl>::MooDeriv(GaugeField &mat, const FermionField &X, const FermionField &Y, int dag)
{
  assert(0);
}

// Derivative parts
template <class Impl>
void WilsonExpCloverFermion<Impl>::MeeDeriv(GaugeField &mat, const FermionField &U, const FermionField &V, int dag)
{
  assert(0); // not implemented yet
}

NAMESPACE_END(Grid);
