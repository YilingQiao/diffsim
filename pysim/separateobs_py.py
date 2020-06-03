import torch
import arcsim

class SeparateObsFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, inp_xold, inp_w, inp_n, zone):
		ans = arcsim.SO_solve_ixns_forward(inp_xold, inp_w, inp_n, zone)
		ctx.intermediate = zone
		ctx.save_for_backward(*ans)
		return ans[0]

	@staticmethod
	def backward(ctx, dldz):
		return tuple(arcsim.SO_solve_ixns_backward(dldz, *(ctx.saved_tensors+(ctx.intermediate,)))+[None])

solve_ixns = SeparateObsFunc.apply
