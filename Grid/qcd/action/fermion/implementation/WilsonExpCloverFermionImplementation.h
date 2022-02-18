/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./lib/qcd/action/fermion/WilsonExpCloverFermion.cc

    Copyright (C) 2017

    Author: paboyle <paboyle@ph.ed.ac.uk>
    Author: Guido Cossu <guido.cossu@ed.ac.uk>

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
  WilsonFermion<Impl>::ImportGauge(_Umu);
  GridBase *grid = _Umu.Grid();
  typename Impl::GaugeLinkField Bx(grid), By(grid), Bz(grid), Ex(grid), Ey(grid), Ez(grid);

  // Compute the field strength terms mu>nu
  WilsonLoops<Impl>::FieldStrength(Bx, _Umu, Zdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(By, _Umu, Zdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Bz, _Umu, Ydir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ex, _Umu, Tdir, Xdir);
  WilsonLoops<Impl>::FieldStrength(Ey, _Umu, Tdir, Ydir);
  WilsonLoops<Impl>::FieldStrength(Ez, _Umu, Tdir, Zdir);

  // Compute the Clover Operator acting on Colour and Spin
  // multiply here by the clover coefficients for the anisotropy
  CloverTerm  = fillCloverYZ(Bx) * csw_r;
  CloverTerm += fillCloverXZ(By) * csw_r;
  CloverTerm += fillCloverXY(Bz) * csw_r;
  CloverTerm += fillCloverXT(Ex) * csw_t;
  CloverTerm += fillCloverYT(Ey) * csw_t;
  CloverTerm += fillCloverZT(Ez) * csw_t;

  //
  // Make Exp Clover and inverse
  //

  int lvol = _Umu.Grid()->lSites();
  int DimRep = Impl::Dimension;
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

  	typename SiteCloverType::scalar_object Qx = Zero(), Qxexp = Zero();

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

  	ExpCloverOp = ExponentiateInternal(srcCloverOpUL,1.0/(diag_mass));

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


  	ExpCloverOp = ExponentiateInternal(srcCloverOpLR,1.0/(diag_mass));

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

  // Add the twisted mass
  CloverFieldType T(CloverTerm.Grid());
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

  {
    autoView(CTExpv,ExpCloverTerm,CpuRead);
    autoView(CTExpIv,ExpCloverTermInv,CpuWrite);
    thread_for(site, lvol, {
      Coordinate lcoor;
      grid->LocalIndexToLocalCoor(site, lcoor);
      Eigen::MatrixXcd EigenCloverOp = Eigen::MatrixXcd::Zero(Ns/2 * DimRep, Ns/2 * DimRep);
      Eigen::MatrixXcd EigenInvCloverOp = Eigen::MatrixXcd::Zero(Ns/2 * DimRep, Ns/2 * DimRep);
      typename SiteCloverType::scalar_object Qx = Zero(), Qxinv = Zero();
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


  // Separate the even and odd parts
  pickCheckerboard(Even, ExpCloverTermEven, ExpCloverTerm);
  pickCheckerboard(Odd, ExpCloverTermOdd, ExpCloverTerm);

  pickCheckerboard(Even, ExpCloverTermDagEven, adj(ExpCloverTerm));
  pickCheckerboard(Odd, ExpCloverTermDagOdd, adj(ExpCloverTerm));

  pickCheckerboard(Even, ExpCloverTermInvEven, ExpCloverTermInv);
  pickCheckerboard(Odd, ExpCloverTermInvOdd, ExpCloverTermInv);

  pickCheckerboard(Even, ExpCloverTermInvDagEven, adj(ExpCloverTermInv));
  pickCheckerboard(Odd, ExpCloverTermInvDagOdd, adj(ExpCloverTermInv));
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
  CloverFieldType *Clover;
  assert(in.Checkerboard() == Odd || in.Checkerboard() == Even);

  if (dag)
  {
    if (in.Grid()->_isCheckerBoarded)
    {
      if (in.Checkerboard() == Odd)
      {
        Clover = (inv) ? &ExpCloverTermInvDagOdd : &ExpCloverTermDagOdd;
      }
      else
      {
        Clover = (inv) ? &ExpCloverTermInvDagEven : &ExpCloverTermDagEven;
      }
      out = *Clover * in;
    }
    else
    {
      Clover = (inv) ? &ExpCloverTermInv : &ExpCloverTerm;
      out = adj(*Clover) * in;
    }
  }
  else
  {
    if (in.Grid()->_isCheckerBoarded)
    {

      if (in.Checkerboard() == Odd)
      {
        //  std::cout << "Calling clover term Odd" << std::endl;
        Clover = (inv) ? &ExpCloverTermInvOdd : &ExpCloverTermOdd;
      }
      else
      {
        //  std::cout << "Calling clover term Even" << std::endl;
        Clover = (inv) ? &ExpCloverTermInvEven : &ExpCloverTermEven;
      }
      out = *Clover * in;
      //  std::cout << GridLogMessage << "*Clover.Checkerboard() "  << (*Clover).Checkerboard() << std::endl;
    }
    else
    {
      Clover = (inv) ? &ExpCloverTermInv : &ExpCloverTerm;
      out = *Clover * in;
    }
  }

} // MooeeInternal

template <class Impl>
iMatrix<ComplexD,6> WilsonExpCloverFermion<Impl>::ExponentiateInternal(const iMatrix<ComplexD,6> &arg, const RealD& alpha)
{
	typedef iMatrix<ComplexD,6> mat;
	int Niter = DEFAULT_MAT_EXP_CLOVER;

	RealD qn[6];
    RealD qnold[6];
	RealD p[5];
	RealD trA2, trA3, trA4;

	mat A2, A3, A4, A5;
	A2 = alpha * alpha * arg * arg;
	A3 = alpha * arg * A2;
	A4 = A2 * A2;
	A5 = A2 * A3;

	trA2 = toReal( trace(A2) );
	trA3 = toReal( trace(A3) );
	trA4 = toReal( trace(A4));

	p[0] = toReal( trace(A3 * A3)) / 6.0 - 0.125 * trA4 * trA2 - trA3 * trA3 / 18.0 + trA2 * trA2 * trA2/ 48.0;
	p[1] = toReal( trace(A5)) / 5.0 - trA3 * trA2 / 6.0;
	p[2] = toReal( trace(A4)) / 4.0 - 0.125 * trA2 * trA2;
	p[3] = trA3 / 3.0;
	p[4] = 0.5 * trA2;

	qnold[0] = cN[Niter];
	qnold[1] = 0.0;
	qnold[2] = 0.0;
	qnold[3] = 0.0;
	qnold[4] = 0.0;
	qnold[5] = 0.0;

	for(int i = Niter-1; i >= 0; i--)
	{
	 qn[0] = p[0] * qnold[5] + cN[i];
     qn[1] = p[1] * qnold[5] + qnold[0];
	 qn[2] = p[2] * qnold[5] + qnold[1];
	 qn[3] = p[3] * qnold[5] + qnold[2];
	 qn[4] = p[4] * qnold[5] + qnold[3];
	 qn[5] = qnold[4];

	 qnold[0] = qn[0];
	 qnold[1] = qn[1];
	 qnold[2] = qn[2];
	 qnold[3] = qn[3];
	 qnold[4] = qn[4];
	 qnold[5] = qn[5];
	}

	mat unit(1.0);

	return (qn[0] * unit + qn[1] * alpha * arg + qn[2] * A2 + qn[3] * A3 + qn[4] * A4 + qn[5] * A5);

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
