import Data.List

main = interact $ (++"\n") . show . length . filter ("f " `isPrefixOf`) . lines
