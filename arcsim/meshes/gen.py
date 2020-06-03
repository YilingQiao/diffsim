num=300
l=1e-1
h=0.5e-1
with open('strip.obj','w') as f:
	for i in range(int(num/2)):
		f.write('v {} {} {}\n'.format(0,i*l,0))
		f.write('v {} {} {}\n'.format(l,i*l,0))
		f.write('v {} {} {}\n'.format(2*l,i*l,0))
		f.write('v {} {} {}\n'.format(0,i*l+h,l))
		f.write('v {} {} {}\n'.format(l,i*l+h,l))
		f.write('v {} {} {}\n'.format(2*l,i*l+h,l))
	for i in range(num-1):
		f.write('f {} {} {}\n'.format(i*3+1,i*3+2,i*3+4))
		f.write('f {} {} {}\n'.format(i*3+4,i*3+2,i*3+5))
		f.write('f {} {} {}\n'.format(i*3+5,i*3+2,i*3+6))
		f.write('f {} {} {}\n'.format(i*3+6,i*3+2,i*3+3))
