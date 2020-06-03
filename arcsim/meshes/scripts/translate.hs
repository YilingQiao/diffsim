import Obj

main = interact $ showObj . map translate . readObj

translate (NX [x,y,z]) = NX [x,y,z+0.02]
translate l = l
