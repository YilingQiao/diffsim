import Obj

main = interact $ showObj . map rotate . readObj

rotate (NX [x,y,z]) = NX [z,y,-x]
rotate l = l
