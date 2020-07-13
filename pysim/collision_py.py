import torch
import time
import arcsim

class CollisionFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx, inp_xold, inp_w, inp_n, zone):

		ans = arcsim.apply_inelastic_projection_forward(inp_xold, inp_w, inp_n, zone)
		ctx.intermediate = zone
		ctx.save_for_backward(*ans)
	
		
		return ans[0]

	@staticmethod
	def backward(ctx, dldz):
		st = time.time()
		ans = tuple(arcsim.apply_inelastic_projection_backward(dldz, *(ctx.saved_tensors+(ctx.intermediate,)))+[None])
		#print("backward----------------------")
		#print(dldz)
		#print(ans)
		#print('collsion backward time={} for nvar={}'.format(time.time()-st,dldz.shape[0]))
		return ans

apply_inelastic_projection = CollisionFunc.apply
