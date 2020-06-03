main = interact $ unlines . map Main.flip . lines

flip ('f':' ':vs) = ("f "++) . unwords . reverse . words $ vs
flip line = line
