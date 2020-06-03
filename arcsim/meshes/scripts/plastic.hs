import Obj
import Data.Maybe (mapMaybe)

main = interact $ showObj . mapMaybe plastic . readObj

plastic (NX _) = Nothing
plastic (NY y) = Just $ NX y
plastic line = Just line
