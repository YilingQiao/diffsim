import Control.Arrow
import Data.List
import Data.Maybe
import Debug.Trace

type Point = (Double, Double)
type Polygon = [Point]

main :: IO ()
main = interact $ writePoly . readSvg

readSvg :: String -> [Polygon]
readSvg = map svgToPoints . findBetween " d=\"m" "\""

findBetween :: String -> String -> String -> [String]
findBetween start end = unfoldr go
  where go s = do
          (_, s') <- splitAround start s
          (w, s'') <- splitAround end s'
          return (w, s'')

splitAround :: Eq a => [a] -> [a] -> Maybe ([a], [a])
splitAround p s = do
  let ss = splits s
  sa <- find ((p `isPrefixOf`) . snd) ss
  let pre = fst sa
  post <- stripPrefix p (snd sa)
  return (pre, post)

splits :: [a] -> [([a],[a])]
splits s = map (\n -> splitAt n s) [0 .. length s]

svgToPoints :: String -> [Point]
svgToPoints = runningSum (0,0) . map svgToPoint . words
  where svgToPoint = (read *** read) . fromJust . splitAround ","
        runningSum _ [] = []
        runningSum p0 (p:ps) = p1 : runningSum p1 ps
          where p1 = (fst p0 + fst p, snd p0 + snd p)

newtype Counter a = Counter (Int -> (Int, a))
instance Monad Counter where
  return x = Counter (\n -> (n,x))
  (Counter c) >>= f = Counter (\n -> let (n',x) = c n; Counter c' = f x in c' n')

get :: Counter Int
get = Counter (\n -> (n, n))

increment :: Counter ()
increment = Counter (\n -> (n+1, ()))

run :: Int -> Counter a -> a
run n (Counter c) = snd (c n)

writePoly :: [Polygon] -> String
writePoly polys = concat [writeVerts (concat ps) (concat polys), writeSegs ps, "0\n"]
  where ps = run 0 (enumerate' polys)

enumerate :: [a] -> Counter [Int]
enumerate [] = return []
enumerate (x:xs) = do n <- get; increment; ns <- enumerate xs; return (n:ns)

enumerate' :: [[a]] -> Counter [[Int]]
enumerate' [] = return []
enumerate' (xs:xss) = do ns <- enumerate xs; nss <- enumerate' xss; return (ns:nss)

counting :: (a -> Counter b) -> [a] -> Counter [b]
counting c [] = return []
counting c (x:xs) = do y <- c x; ys <- counting c xs; return (y:ys)

incrementing :: (Int -> a -> b) -> a -> Counter b
incrementing f x = do n <- get; let {y = f n x}; increment; return y

writeVerts :: [Int] -> [Point] -> String
writeVerts ps points = unlines . (header:) $ zipWith vert ps points
  where vert n (x,y) = unwords [show n, show x, show y]
        header = unwords [show . length $ ps, "2", "0", "0"]

writeSegs :: [[Int]] -> String
writeSegs pss = unlines . (header:) $ run 0 (go segs)
  where go = counting (incrementing (\n (x,y) -> unwords [show n, show x, show y]))
        header = unwords [show . length . concat $ pss, "0"]
        rotate xs = tail xs ++ [head xs]
        segs = concatMap (\ps -> zip ps (rotate ps)) pss

trace' :: Show a => a -> a
trace' x = trace (show x) x
