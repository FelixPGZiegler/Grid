/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/qcd/action/fermion/CompactWilsonExpCloverFermionImplementation.h

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
#include <Grid/qcd/action/fermion/CompactWilsonExpCloverFermion.h>

NAMESPACE_BEGIN(Grid);
template<class Impl>
CompactWilsonExpCloverFermion<Impl>::CompactWilsonExpCloverFermion(GaugeField& _Umu,
                                                             GridCartesian& Fgrid,
                                                             GridRedBlackCartesian& Hgrid,
                                                             const RealD _mass,
															 const RealD _twmass,
                                                             const RealD _csw_r,
                                                             const RealD _csw_t,
                                                             const RealD _cF,
                                                             const WilsonAnisotropyCoefficients& clover_anisotropy,
                                                             const ImplParams& impl_p)
  : WilsonBase(_Umu, Fgrid, Hgrid, _mass, impl_p, clover_anisotropy)
  , csw_r(_csw_r)
  , csw_t(_csw_t)
  , cF(_cF)
  , open_boundaries(impl_p.boundary_phases[Nd-1] == 0.0)
  , Diagonal(&Fgrid),        Triangle(&Fgrid)
  , DiagonalExp(&Fgrid),        TriangleExp(&Fgrid)
  , DiagonalExpEven(&Hgrid),    TriangleExpEven(&Hgrid)
  , DiagonalExpOdd(&Hgrid),     TriangleExpOdd(&Hgrid)
  , DiagonalExpInv(&Fgrid),     TriangleExpInv(&Fgrid)
  , DiagonalExpInvEven(&Hgrid), TriangleExpInvEven(&Hgrid)
  , DiagonalExpInvOdd(&Hgrid),  TriangleExpInvOdd(&Hgrid)
  , Tmp(&Fgrid)
  , BoundaryMask(&Fgrid)
  , BoundaryMaskEven(&Hgrid), BoundaryMaskOdd(&Hgrid)
{
  csw_r *= 0.5;
  csw_t *= 0.5;
  if (clover_anisotropy.isAnisotropic)
    csw_r /= clover_anisotropy.xi_0;

  twmass = _twmass;

  ImportGauge(_Umu);
  if (open_boundaries)
    CompactHelpers::SetupMasks(this->BoundaryMask, this->BoundaryMaskEven, this->BoundaryMaskOdd);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::Dhop(const FermionField& in, FermionField& out, int dag) {
  WilsonBase::Dhop(in, out, dag);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::DhopOE(const FermionField& in, FermionField& out, int dag) {
  WilsonBase::DhopOE(in, out, dag);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::DhopEO(const FermionField& in, FermionField& out, int dag) {
  WilsonBase::DhopEO(in, out, dag);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::DhopDir(const FermionField& in, FermionField& out, int dir, int disp) {
  WilsonBase::DhopDir(in, out, dir, disp);
  if(this->open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::DhopDirAll(const FermionField& in, std::vector<FermionField>& out) {
  WilsonBase::DhopDirAll(in, out);
  if(this->open_boundaries) {
    for(auto& o : out) ApplyBoundaryMask(o);
  }
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::M(const FermionField& in, FermionField& out) {
  out.Checkerboard() = in.Checkerboard();
  WilsonBase::Dhop(in, out, DaggerNo); // call base to save applying bc
  Mooee(in, Tmp);
  axpy(out, 1.0, out, Tmp);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::Mdag(const FermionField& in, FermionField& out) {
  out.Checkerboard() = in.Checkerboard();
  WilsonBase::Dhop(in, out, DaggerYes);  // call base to save applying bc
  MooeeDag(in, Tmp);
  axpy(out, 1.0, out, Tmp);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::Meooe(const FermionField& in, FermionField& out) {
  WilsonBase::Meooe(in, out);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MeooeDag(const FermionField& in, FermionField& out) {
  WilsonBase::MeooeDag(in, out);
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::Mooee(const FermionField& in, FermionField& out) {
  if(in.Grid()->_isCheckerBoarded) {
    if(in.Checkerboard() == Odd) {
      MooeeInternal(in, out, DiagonalExpOdd, TriangleExpOdd);
    } else {
      MooeeInternal(in, out, DiagonalExpEven, TriangleExpEven);
    }
  } else {
    MooeeInternal(in, out, DiagonalExp, TriangleExp);
  }
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MooeeDag(const FermionField& in, FermionField& out) {
  Mooee(in, out); // blocks are hermitian
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MooeeInv(const FermionField& in, FermionField& out) {
  if(in.Grid()->_isCheckerBoarded) {
    if(in.Checkerboard() == Odd) {
      MooeeInternal(in, out, DiagonalExpInvOdd, TriangleExpInvOdd);
    } else {
      MooeeInternal(in, out, DiagonalExpInvEven, TriangleExpInvEven);
    }
  } else {
    MooeeInternal(in, out, DiagonalExpInv, TriangleExpInv);
  }
  if(open_boundaries) ApplyBoundaryMask(out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MooeeInvDag(const FermionField& in, FermionField& out) {
  MooeeInv(in, out); // blocks are hermitian
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::Mdir(const FermionField& in, FermionField& out, int dir, int disp) {
  DhopDir(in, out, dir, disp);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MdirAll(const FermionField& in, std::vector<FermionField>& out) {
  DhopDirAll(in, out);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MDeriv(GaugeField& force, const FermionField& X, const FermionField& Y, int dag) {
  assert(!open_boundaries); // TODO check for changes required for open bc

  // NOTE: code copied from original clover term
  conformable(X.Grid(), Y.Grid());
  conformable(X.Grid(), force.Grid());
  GaugeLinkField force_mu(force.Grid()), lambda(force.Grid());
  GaugeField clover_force(force.Grid());
  PropagatorField Lambda(force.Grid());

  // Guido: Here we are hitting some performance issues:
  // need to extract the components of the DoubledGaugeField
  // for each call
  // Possible solution
  // Create a vector object to store them? (cons: wasting space)
  std::vector<GaugeLinkField> U(Nd, this->Umu.Grid());

  Impl::extractLinkField(U, this->Umu);

  force = Zero();
  // Derivative of the Wilson hopping term
  this->DhopDeriv(force, X, Y, dag);

  ///////////////////////////////////////////////////////////
  // Clover term derivative
  ///////////////////////////////////////////////////////////
  Impl::outerProductImpl(Lambda, X, Y);
  //std::cout << "Lambda:" << Lambda << std::endl;

  Gamma::Algebra sigma[] = {
      Gamma::Algebra::SigmaXY,
      Gamma::Algebra::SigmaXZ,
      Gamma::Algebra::SigmaXT,
      Gamma::Algebra::MinusSigmaXY,
      Gamma::Algebra::SigmaYZ,
      Gamma::Algebra::SigmaYT,
      Gamma::Algebra::MinusSigmaXZ,
      Gamma::Algebra::MinusSigmaYZ,
      Gamma::Algebra::SigmaZT,
      Gamma::Algebra::MinusSigmaXT,
      Gamma::Algebra::MinusSigmaYT,
      Gamma::Algebra::MinusSigmaZT};

  /*
    sigma_{\mu \nu}=
    | 0         sigma[0]  sigma[1]  sigma[2] |
    | sigma[3]    0       sigma[4]  sigma[5] |
    | sigma[6]  sigma[7]     0      sigma[8] |
    | sigma[9]  sigma[10] sigma[11]   0      |
  */

  int count = 0;
  clover_force = Zero();
  for (int mu = 0; mu < 4; mu++)
  {
    force_mu = Zero();
    for (int nu = 0; nu < 4; nu++)
    {
      if (mu == nu)
        continue;

      RealD factor;
      if (nu == 4 || mu == 4)
      {
        factor = 2.0 * csw_t;
      }
      else
      {
        factor = 2.0 * csw_r;
      }
      PropagatorField Slambda = Gamma(sigma[count]) * Lambda; // sigma checked
      Impl::TraceSpinImpl(lambda, Slambda);                   // traceSpin ok
      force_mu -= factor*Helpers::Cmunu(U, lambda, mu, nu);   // checked
      count++;
    }

    pokeLorentz(clover_force, U[mu] * force_mu, mu);
  }
  //clover_force *= csw;
  force += clover_force;
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MooDeriv(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) {
  assert(0);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MeeDeriv(GaugeField& mat, const FermionField& U, const FermionField& V, int dag) {
  assert(0);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::MooeeInternal(const FermionField&        in,
                    FermionField&              out,
                    const CloverDiagonalField& diagonal,
                    const CloverTriangleField& triangle) {
  assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);
  out.Checkerboard() = in.Checkerboard();
  conformable(in, out);
  conformable(in, diagonal);
  conformable(in, triangle);

  CompactHelpers::MooeeKernel(diagonal.oSites(), 1, in, out, diagonal, triangle);
}

template<class Impl>
void CompactWilsonExpCloverFermion<Impl>::ImportGauge(const GaugeField& _Umu) {
  // NOTE: parts copied from original implementation

  // Import gauge into base class
  double t0 = usecond();
  WilsonBase::ImportGauge(_Umu); // NOTE: called here and in wilson constructor -> performed twice, but can't avoid that

  // Initialize temporary variables
  double t1 = usecond();
  conformable(_Umu.Grid(), this->GaugeGrid());
  GridBase* grid = _Umu.Grid();
  typename Impl::GaugeLinkField Bx(grid), By(grid), Bz(grid), Ex(grid), Ey(grid), Ez(grid);
  CloverField TmpOriginal(grid);

  // Compute the field strength terms mu>nu
  double t2 = usecond();
  WilsonLoops<Impl>::FieldStrength(Bx, _Umu, Zdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(By, _Umu, Zdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Bz, _Umu, Ydir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ex, _Umu, Tdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ey, _Umu, Tdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(Ez, _Umu, Tdir, Zdir);

  // Compute the Clover Operator acting on Colour and Spin
  // multiply here by the clover coefficients for the anisotropy
  double t3 = usecond();
  TmpOriginal  = Helpers::fillCloverYZ(Bx) * csw_r;
  TmpOriginal += Helpers::fillCloverXZ(By) * csw_r;
  TmpOriginal += Helpers::fillCloverXY(Bz) * csw_r;
  TmpOriginal += Helpers::fillCloverXT(Ex) * csw_t;
  TmpOriginal += Helpers::fillCloverYT(Ey) * csw_t;
  TmpOriginal += Helpers::fillCloverZT(Ez) * csw_t;

  // Convert the data layout of the clover term
  double t4 = usecond();
  CompactHelpers::ConvertLayout(TmpOriginal, Diagonal, Triangle);

  // Possible modify the boundary values
  double t5 = usecond();
  if(open_boundaries) CompactHelpers::ModifyBoundaries(Diagonal, Triangle, csw_t, cF, this->diag_mass);

  // Exponentiate

  double t6 = usecond();
  CompactHelpers::Exponentiate(Diagonal,
                               Triangle,
                               DiagonalExp,
                               TriangleExp,
                               1.0 / this->diag_mass);

  // ToDo: Does this work?
  DiagonalExp *= this->diag_mass;
  TriangleExp *= this->diag_mass;

  // Add twisted mass
  typedef typename SiteCloverDiagonal::scalar_object scalar_object_diagonal;

  double t7 = usecond();
  long lsites = DiagonalExp.Grid()->lSites();
  autoView(T_v,  DiagonalExp,  CpuRead);
  autoView(DiagonalExp_v, DiagonalExp, CpuWrite);

  thread_for(site, lsites,
  {
    scalar_object_diagonal diagonal_src = Zero();
    scalar_object_diagonal diagonal_dst = Zero();

    Coordinate lcoor;
    grid->LocalIndexToLocalCoor(site, lcoor);

    peekLocalSite(diagonal_src, T_v, lcoor);
    ComplexD sum;

    for(int i = 0; i < 6; i++){
      sum = static_cast<ComplexD>(TensorRemove(diagonal_src()(0)(i))) + timesI(static_cast<ComplexD>(twmass));
      diagonal_dst()(0)(i) = sum;

      sum = static_cast<ComplexD>(TensorRemove(diagonal_src()(1)(i))) - timesI(static_cast<ComplexD>(twmass));
      diagonal_dst()(1)(i) = sum;
    }

   pokeLocalSite(diagonal_dst, DiagonalExp_v, lcoor);

  });

  // Invert the clover term in the improved layout
  double t8 = usecond();
  CompactHelpers::Invert(DiagonalExp, TriangleExp, DiagonalExpInv, TriangleExpInv);

  // Fill the remaining clover fields
  double t9 = usecond();
  pickCheckerboard(Even, DiagonalExpEven,    DiagonalExp);
  pickCheckerboard(Even, TriangleExpEven,    TriangleExp);
  pickCheckerboard(Odd,  DiagonalExpOdd,     DiagonalExp);
  pickCheckerboard(Odd,  TriangleExpOdd,     TriangleExp);
  pickCheckerboard(Even, DiagonalExpInvEven, DiagonalExpInv);
  pickCheckerboard(Even, TriangleExpInvEven, TriangleExpInv);
  pickCheckerboard(Odd,  DiagonalExpInvOdd,  DiagonalExpInv);
  pickCheckerboard(Odd,  TriangleExpInvOdd,  TriangleExpInv);

  // Report timings
  double t10 = usecond();
#if 0
  std::cout << GridLogMessage << "CompactWilsonExpCloverFermion::ImportGauge timings:"
            << " WilsonFermion::Importgauge = " << (t1 - t0) / 1e6
            << ", allocations = "               << (t2 - t1) / 1e6
            << ", field strength = "            << (t3 - t2) / 1e6
            << ", fill clover = "               << (t4 - t3) / 1e6
            << ", convert = "                   << (t5 - t4) / 1e6
            << ", boundaries = "                << (t6 - t5) / 1e6
			<< ", exponentiation = "                << (t7 - t6) / 1e6
			<< ", twisted mass = "                << (t8 - t7) / 1e6
            << ", inversions = "                << (t9 - t8) / 1e6
            << ", pick cbs = "                  << (t10 - t9) / 1e6
            << ", total = "                     << (t10 - t0) / 1e6
            << std::endl;
#endif
}

NAMESPACE_END(Grid);
