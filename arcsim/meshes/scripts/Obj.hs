module Obj (Obj, ObjLine(..), readObj, showObj) where

import Data.List (intercalate)
import Data.List.Split (splitOn)
import Data.Maybe (mapMaybe)

data ObjLine = VU [Double] | VL Int
             | NX [Double] | NY [Double] | NV [Double] | NL Int
             | E [Int] | EA Double | EL Int
             | F [[Int]] | FL Int
             deriving Show

type Obj = [ObjLine]

readObj :: String -> Obj
readObj = mapMaybe readLine . lines

readLine :: String -> Maybe ObjLine
readLine ('v':'t':' ':us) = Just . VU . map read . words $ us
readLine ('v':'l':' ':l) = Just . VL . read $ l
readLine ('v':' ':xs) = Just . NX . map read . words $ xs
readLine ('n':'y':' ':ys) = Just . NY . map read . words $ ys
readLine ('n':'v':' ':vs) = Just . NV . map read . words $ vs
readLine ('n':'l':' ':l) = Just . NL . read $ l
readLine ('e':' ':vs) = Just . E . map read . words $ vs
readLine ('e':'a':' ':a) = Just . EA . read $ a
readLine ('e':'l':' ':l) = Just . EL . read $ l
readLine ('f':' ':vs) = Just . F . map readTriplet . words $ vs
readLine ('f':'l':' ':l) = Just . FL . read $ l
readLine _ = Nothing

readTriplet :: String -> [Int]
readTriplet = map read . splitOn "/"

showObj :: Obj -> String
showObj = unlines . map showLine

showLine :: ObjLine -> String
showLine (VU us) = ("vt "++) . unwords . map show $ us
showLine (VL l) = ("vl "++) . show $ l
showLine (NX xs) = ("v "++) . unwords . map show $ xs
showLine (NY ys) = ("ny "++) . unwords . map show $ ys
showLine (NV vs) = ("nv "++) . unwords . map show $ vs
showLine (NL l) = ("nl "++) . show $ l
showLine (E vs) = ("e "++) . unwords . map show $ vs
showLine (EA a) = ("ea "++) . show $ a
showLine (EL l) = ("el "++) . show $ l
showLine (F vs) = ("f "++) . unwords . map showTriplet $ vs
showLine (FL l) = ("fl "++) . show $ l

showTriplet :: [Int] -> String
showTriplet = intercalate "/" . map show

-- if (keyword == "vt") {
--     Vec2 u;
--     linestream >> u[0] >> u[1];
--     mesh.add(new Vert(u));
-- } else if (keyword == "vl") {
--     linestream >> mesh.verts.back()->label;
-- } else if (keyword == "v") {
--     Vec3 x;
--     linestream >> x[0] >> x[1] >> x[2];
--     mesh.add(new Node(x, Vec3(0)));
-- } else if (keyword == "ny") {
--     Vec3 &y = mesh.nodes.back()->y;
--     linestream >> y[0] >> y[1] >> y[2];
-- } else if (keyword == "nv") {
--     Vec3 &v = mesh.nodes.back()->v;
--     linestream >> v[0] >> v[1] >> v[2];
-- } else if (keyword == "nl") {
--     linestream >> mesh.nodes.back()->label;
-- } else if (keyword == "e") {
--     int v0, v1;
--     linestream >> v0 >> v1;
--     mesh.add(new Edge(mesh.verts[v0-1], mesh.verts[v1-1]));
-- } else if (keyword == "ea") {
--     linestream >> mesh.edges.back()->theta_ideal;
-- } else if (keyword == "el") {
--     linestream >> mesh.edges.back()->label;
-- } else if (keyword == "f") {
--     vector<Vert*> verts;
--     vector<Node*> nodes;
--     string w;
--     while (linestream >> w) {
--         stringstream wstream(w);
--         int v, n;
--         char c;
--         wstream >> n >> c >> v;
--         nodes.push_back(mesh.nodes[n-1]);
--         if (wstream)
--             verts.push_back(mesh.verts[v-1]);
--         else if (!nodes.back()->verts.empty())
--             verts.push_back(nodes.back()->verts[0]);
--         else {
--             verts.push_back(new Vert(project<2>(nodes.back()->x)));
--             mesh.add(verts.back());
--         }
--     }
--     for (int v = 0; v < verts.size(); v++)
--         connect(verts[v], nodes[v]);
--     vector<Face*> faces = triangulate(verts);
--     for (int f = 0; f < faces.size(); f++)
--         mesh.add(faces[f]);
-- } else if (keyword == "fl") {
--     linestream >> mesh.faces.back()->label;
-- }
