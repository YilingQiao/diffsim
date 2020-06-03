import Control.Monad
import Obj
import System.Environment

main = do
  args <- liftM (map read) getArgs
  interact $ showObj . map (fold args) . readObj

fold [x0,y0,x1,y1,theta] (NX [x,y,z]) = NX $ rotate (x0,y0) (x1,y1) theta (x,y) z
fold _ line = line

rotate p0@(x0,y0) p1@(x1,y1) theta p@(x,y) z =
  let dp = (x-x0, y-y0)
      u = normalize (x1-x0, y1-y0)
      v = perp u
      pu = p <.> u
      pv = p <.> v
      rad = theta*pi/180
      p2@(x2,y2) = (pu <*> u) <+> ((pv * cos rad) <*> v)
  in
   if pv < 0
   then [x, y, z]
   else [x2, y2, pv * sin rad]

normalize (x, y) = (x/n, y/n) where n = sqrt (x*x + y*y)
perp (x, y) = (-y, x)
(x1,y1) <.> (x2,y2) = x1*x2 + y1*y2
a <*> (x,y) = (a*x, a*y)
(x1,y1) <+> (x2,y2) = (x1+x2, y1+y2)
