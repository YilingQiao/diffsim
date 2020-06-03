import torch
import arcsim

class CubicFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, a,b,c,d):
		ans = arcsim.solve_cubic_forward(a,b,c,d)
		ctx.save_for_backward(ans,a,b,c,d)
		return ans

	@staticmethod
	def backward(ctx, dldz):
		return tuple(arcsim.solve_cubic_backward(dldz, *ctx.saved_tensors))

solve_cubic = CubicFunc.apply
