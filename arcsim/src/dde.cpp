/*
  Copyright ©2013 The Regents of the University of California
  (Regents). All Rights Reserved. Permission to use, copy, modify, and
  distribute this software and its documentation for educational,
  research, and not-for-profit purposes, without fee and without a
  signed licensing agreement, is hereby granted, provided that the
  above copyright notice, this paragraph and the following two
  paragraphs appear in all copies, modifications, and
  distributions. Contact The Office of Technology Licensing, UC
  Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620,
  (510) 643-7201, for commercial licensing opportunities.

  IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT,
  INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
  LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
  DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY
  OF SUCH DAMAGE.

  REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING
  DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS
  IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
  UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

#include "dde.hpp"

#include "cloth.hpp"
#include "util.hpp"

using namespace std;
using torch::Tensor;

static const int nsamples = 30;

Tensor evaluate_stretching_sample (const Tensor &G, const StretchingData &data);

void evaluate_stretching_samples (StretchingSamples &samples,
                                  const StretchingData &data) {
  vector<double> inpgrid;
    for(int i = 0; i < ::nsamples; i++)
        for(int j = 0; j < ::nsamples; j++)
            for(int k = 0; k < ::nsamples; k++)
                {
                    double a,b,c;
                    a=0.5+i*2/(::nsamples*1.0);
                    b=0.5+j*2/(::nsamples*1.0);
                    c=k*2/(::nsamples*1.0);
                    // samples.s[i][j][k] = evaluate_stretching_sample(G, data);
                    double w = 0.5*(a+b+sqrt(4*sq(c)+sq(a-b)));
                    double v1 = c, v0 = w - b;
                    if (k == 0)
                      if (i >= j) {
                        v1=0;v0=1;
                      } else {
                        v1=1;v0=0;
                      }
                    double angle_weight = fabs(atan2(v1,v0)/M_PI)*8;
                    double strain_weight = (sqrt(w)-1)*6;
                    inpgrid.push_back(angle_weight/2-1);
                    inpgrid.push_back(strain_weight*2-1);
                }
  Tensor grid = torch::tensor(inpgrid,TNOPT).reshape({1,nsamples*nsamples,nsamples,2});
  samples = torch::grid_sampler(data, grid, 0, 1, 0).reshape({1,4,nsamples,nsamples,nsamples});
  samples = torch::relu(samples*2);
}

Tensor stretching_stiffness (const Tensor &G, const StretchingSamples &samples) {
    Tensor a=(G[0]+0.25);
    Tensor b=(G[3]+0.25);
    Tensor c=abs(G[1]);
    Tensor grid = torch::stack({c,b,a}).reshape({1,1,1,1,3})*(nsamples*2/(nsamples-1.))-1;
    Tensor stiffness = torch::grid_sampler(samples, grid, 0, 1, 0).squeeze();
    return stiffness;
}

Tensor batch_stretching_stiffness (const Tensor &G, const Tensor &samples) {
    Tensor a=(G[0]+0.25);
    Tensor b=(G[3]+0.25);
    Tensor c=abs(G[1]);
    Tensor grid = torch::stack({c,b,a}, 1).reshape({1,1,1,-1,3})*(nsamples*2/(nsamples-1.))-1;
    Tensor stiffness = torch::grid_sampler(samples.squeeze(0), grid, 0, 1, 0).squeeze();//.t();//4xn
    return stiffness;
}

Tensor bending_stiffness (const Edge *edge, const BendingData &data0, const BendingData &data1) {
    Tensor curv = edge->theta*edge->ldaa*0.05;//l/(edge->adjf[0]->a + edge->adjf[1]->a);
    Tensor value = clamp(curv-1, -1, 1); // because samples are per 0.05 cm^-1 = 5 m^-1
    //0
    Tensor    bias_angle0=edge->bias_angle[0];//(atan2(du0[1], du0[0]))*(4/M_PI);
    Tensor grid0 = torch::stack({value, bias_angle0}).reshape({1,1,1,2});
    Tensor actual_ke0 = relu(grid_sampler(data0, grid0, 0, 2, 0).squeeze());
    //1
    Tensor    bias_angle1=edge->bias_angle[1];//(atan2(du1[1], du1[0]))*(4/M_PI);
    if ((data0==data1).all().item<int>() && (bias_angle0==bias_angle1).all().item<int>())
      return actual_ke0;
    Tensor grid1 = torch::stack({value, bias_angle1}).reshape({1,1,1,2});
    Tensor actual_ke1 = relu(grid_sampler(data1, grid1, 0, 2, 0).squeeze());
    return min(actual_ke0, actual_ke1);
}

Tensor batch_bending_stiffness(Tensor curv, Tensor bang, Tensor bend) {
    Tensor value = clamp(curv-1, -1, 1); // because samples are per 0.05 cm^-1 = 5 m^-1
    bend = bend.squeeze(0);
    //0
    Tensor    bias_angle0=bang[0];//(atan2(du0[1], du0[0]))*(4/M_PI);
    Tensor grid0 = torch::stack({value, bias_angle0}, 1).reshape({1,1,-1,2});
    Tensor actual_ke0 = relu(grid_sampler(bend, grid0, 0, 2, 0).squeeze());
    //1
    Tensor    bias_angle1=bang[1];//(atan2(du1[1], du1[0]))*(4/M_PI);
    if ((bias_angle0==bias_angle1).all().item<int>())
      return actual_ke0;
    Tensor grid1 = torch::stack({value, bias_angle1}, 1).reshape({1,1,-1,2});
    Tensor actual_ke1 = relu(grid_sampler(bend, grid1, 0, 2, 0).squeeze());
    return min(actual_ke0, actual_ke1);
}

