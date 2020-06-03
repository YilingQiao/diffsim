import torch
import arcsim

class SeparateFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, inp_xold, inp_w, inp_n, zone):
		ans = arcsim.solve_ixns_forward(inp_xold, inp_w, inp_n, zone)
		ctx.intermediate = zone
		ctx.save_for_backward(*ans)
		return ans[0]

	@staticmethod
	def backward(ctx, dldz):
		return tuple(arcsim.solve_ixns_backward(dldz, *(ctx.intermediate+(ctx.intermediate,)))+[None])

solve_ixns = SeparateFunc.apply
