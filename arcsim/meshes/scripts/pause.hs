import Control.Monad
import Data.List
import System.Environment

type Vec = [Double]
data Block = Block String [Vec]
data Mot = Mot Int [Block]

main :: IO ()
main = do
  [sf, pf] <- liftM (map read) getArgs
  interact $ unlines . writeMot . pauseMot pf . shiftMot sf . readMot . lines

readMot :: [String] -> Mot
readMot (l:ls) = Mot (readFrames l) (readBlocks ls)

readFrames :: String -> Int
readFrames = read . (!! 1) . words

readBlocks :: [String] -> [Block]
readBlocks = map readBlock . split ""

split :: Eq a => a -> [a] -> [[a]]
split a [] = [[]]
split a (b:bs) | a==b = [] : split a bs
split a (b:bs) = let ss = split a bs in (b : head ss) : tail ss

readBlock :: [String] -> Block
readBlock (h:d) = Block h (readData d)

readData :: [String] -> [Vec]
readData = map (map read . words)

writeMot :: Mot -> [String]
writeMot (Mot f bs) = writeFrames f : writeBlocks bs

writeFrames :: Int -> String
writeFrames f = "NumFrames: " ++ show f

writeBlocks :: [Block] -> [String]
writeBlocks = intercalate [""] . map writeBlock

writeBlock :: Block -> [String]
writeBlock (Block h d) = h : writeData d

writeData :: [Vec] -> [String]
writeData = map (unwords . map show)

shiftMot :: Int -> Mot -> Mot
shiftMot n (Mot f bs) = Mot (f - n) (map (shiftBlock n) bs)

shiftBlock :: Int -> Block -> Block
shiftBlock n (Block h d) = Block h (drop n d)

pauseMot :: Int -> Mot -> Mot
pauseMot n (Mot f bs) = Mot (f + n) (map (pauseBlock n) bs)

pauseBlock :: Int -> Block -> Block
pauseBlock n (Block h d) = Block h (replicate n (head d) ++ d)
