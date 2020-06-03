import Obj

main = interact $ showObj . concatMap label . map roll . readObj

roll (NX [x,y,z]) = NX [r*cos theta, r*sin theta, y]
  where theta = 2*pi*x/c
roll line = line

c = 0.280 -- circumference
s = 1 -- scaling factor
r = s*c/(2*pi)

label (NX [x,y,z]) | z < 0+eps = [NX [x,y,z], NL 1]
label (NX [x,y,z]) | z > h-eps = [NX [x,y,z], NL 2]
label line = [line]

h = 0.216 -- height
eps = 1e-3
