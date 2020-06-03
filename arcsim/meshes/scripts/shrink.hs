main = interact $ unlines . map shrink . lines

shrink ('v':' ':xs) = ("v "++) . unwords . map (show . (/100) . read) . words $ xs
shrink line = line
