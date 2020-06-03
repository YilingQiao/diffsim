import torch
import time
import arcsim

class TaucsFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, A,b, ind):
		#print(A)
		#print(b)
		#print(ind)
		ans = arcsim.taucs_linear_solve_forward(A,b,ind)
		ctx.indi = ind;
		ctx.save_for_backward(ans,A,b)
		return ans

	@staticmethod
	def backward(ctx, dldz):
		st = time.time()
		ans = tuple(arcsim.taucs_linear_solve_backward(dldz, ctx.indi, *ctx.saved_tensors)+[None])
		#print(dldz)
		#print(ans[0])
		#print(ans[1])
		# print('taucs backard time={}\n'.format(time.time()-st))
		return ans

taucs_linear_solve = TaucsFunc.apply
