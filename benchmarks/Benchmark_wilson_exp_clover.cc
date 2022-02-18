    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./benchmarks/Benchmark_wilson_exp_clover.cc

    Copyright (C) 2018

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: paboyle <paboyle@ph.ed.ac.uk>

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

using namespace std;
using namespace Grid;
 ;


#include "Grid/util/Profiling.h"

template<class d>
struct scal {
  d internal;
};

bool perfProfiling = false;

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  if( GridCmdOptionExists(argv,argv+argc,"--perf") ){
    perfProfiling = true;
  }

  long unsigned int single_site_flops = 1920; // number used in openQCD. arXiv:1412.2629 states 1848.

  auto latt_size   = GridDefaultLatt();
  auto simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  auto mpi_layout  = GridDefaultMpi();

  GridCartesian               Grid(latt_size,simd_layout,mpi_layout);
  GridRedBlackCartesian     RBGrid(&Grid);

  int threads = GridThread::GetThreads();

  GridLogLayout();

  std::cout<<GridLogMessage << "Grid floating point word size is REALF"<< sizeof(RealF)<<std::endl;
  std::cout<<GridLogMessage << "Grid floating point word size is REALD"<< sizeof(RealD)<<std::endl;
  std::cout<<GridLogMessage << "Grid floating point word size is REAL"<< sizeof(Real)<<std::endl;
  std::cout<<GridLogMessage << "Grid number of colours : "<< Nc <<std::endl;
  std::cout<<GridLogMessage << "Benchmarking Wilson Exponential Clover operator in the fundamental representation" << std::endl;


  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&Grid);
  pRNG.SeedFixedIntegers(seeds);

  LatticeFermion src   (&Grid); random(pRNG,src);
  LatticeFermion result(&Grid); result=Zero();
  LatticeFermion    tmp(&Grid);    tmp=Zero();
  LatticeGaugeField Umu(&Grid); random(pRNG,Umu);

  double volume=1;
  for(int mu=0;mu<Nd;mu++){
    volume=volume*latt_size[mu];
  }

  RealD mass=0.1;

  typename CompactWilsonExpCloverFermionR::ImplParams params;
  WilsonAnisotropyCoefficients anis;

  // CompactWilsonExpCloverFermion with csw=1
  CompactWilsonExpCloverFermionR Dw(Umu, Grid, RBGrid, mass, 1.0, 1.0, 1.0, anis, params);

  // 10 warm up calls
  for(int i=0;i<10;i++){
    Dw.M(src,result);
  }

  std::cout<<GridLogMessage << "Calling Dw"<<std::endl;
  int ncall=2000;

  // Counters
  Dw.ZeroCounters();
  Grid.Barrier();

  double t0=usecond();
  for(int i=0;i<ncall;i++){
    Dw.M(src,result);
  }

  // Counters
  Grid.Barrier();

  double t1=usecond();
  double flops=single_site_flops*volume*ncall;

  if (perfProfiling){
  std::cout<<GridLogMessage << "Profiling Dw with perf"<<std::endl;

  System::profile("kernel", [&]() {
    for(int i=0;i<ncall;i++){
      Dw.M(src,result);
    }
  });

  std::cout<<GridLogMessage << "Generated kernel.data"<<std::endl;
  std::cout<<GridLogMessage << "Use with: perf report -i kernel.data"<<std::endl;

  }

  auto nsimd = vComplex::Nsimd();
  auto simdwidth = sizeof(vComplex);

  std::cout<<GridLogMessage << "Nsimd "<< nsimd << std::endl;
  std::cout<<GridLogMessage << "Simd width "<< simdwidth << std::endl;

  // RF: Nd Wilson, Nd gauge, Nc colors
  // double data = volume * ((2*Nd+1)*Nd*Nc + 2*Nd*Nc*Nc) * simdwidth / nsimd * ncall / (1024.*1024.*1024.);

  std::cout<<GridLogMessage << "Called Dw"<<std::endl;
  std::cout<<GridLogMessage << "flops per site " << single_site_flops << std::endl;
  std::cout<<GridLogMessage << "mflop/s =   "<< flops/(t1-t0)<<std::endl;
  // std::cout<<GridLogMessage << "RF  GiB/s (base 2) =   "<< 1000000. * data/(t1-t0)<<std::endl;

  Grid_finalize();
}
