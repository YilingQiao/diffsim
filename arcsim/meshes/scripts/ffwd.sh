#!/bin/bash

runhaskell fold.hs 0 280 0 0 90 < ../ffwd0.obj > ../ffwd1.obj
runhaskell fold.hs 0 280 0 0 180 < ../ffwd0.obj > ../ffwd2.obj
runhaskell fold.hs 108 108 0 0 90 < ../ffwd0.obj > ~/tmp/ffwd.obj
runhaskell fold.hs 0 0 -108 108 90 < ~/tmp/ffwd.obj > ../ffwd3.obj
runhaskell fold.hs 108 108 0 0 180 < ../ffwd0.obj > ~/tmp/ffwd.obj
runhaskell fold.hs 0 0 -108 108 180 < ~/tmp/ffwd.obj > ../ffwd4.obj
runhaskell fold.hs 108 261 0 0 90 < ../ffwd4.obj > ~/tmp/ffwd.obj
runhaskell fold.hs 0 0 -108 261 90 < ~/tmp/ffwd.obj > ../ffwd5.obj
runhaskell fold.hs 108 261 0 0 180 < ../ffwd4.obj > ~/tmp/ffwd.obj
runhaskell fold.hs 0 0 -108 261 180 < ~/tmp/ffwd.obj > ../ffwd6.obj
runhaskell fold.hs 0 280 0 0 90 < ../ffwd6.obj > ../ffwd7.obj
runhaskell fold.hs 0 280 0 0 180 < ../ffwd6.obj > ../ffwd8.obj
runhaskell fold.hs 0 0 -56 280 90 < ../ffwd8.obj > ../ffwd9.obj
runhaskell fold.hs 0 0 -56 280 180 < ../ffwd8.obj > ../ffwd10.obj
$EDITOR ../ffwd8.obj ../ffwd9.obj ../ffwd10.obj
runhaskell fold.hs 0 0 0 280 90 < ../ffwd10.obj > ~/tmp/ffwd.obj
runhaskell translate.hs < ~/tmp/ffwd.obj > ../ffwd11.obj
runhaskell fold.hs 0 0 0 280 180 < ../ffwd10.obj > ../ffwd12.obj
runhaskell fold.hs 56 280 0 0 90 < ../ffwd12.obj > ../ffwd13.obj
runhaskell fold.hs 56 280 0 0 180 < ../ffwd12.obj > ../ffwd14.obj
runhaskell fold.hs 0 280 0 0 90 < ../ffwd14.obj > ../ffwd15.obj
runhaskell fold.hs 0 0 -56 280 45 < ../ffwd8.obj > ~/tmp/ffwd.obj
runhaskell rotate.hs < ~/tmp/ffwd.obj > ../ffwd16.obj
runhaskell fold.hs 0 0 -56 280 90 < ../ffwd8.obj > ~/tmp/ffwd.obj
runhaskell rotate.hs < ~/tmp/ffwd.obj > ../ffwd17.obj
$EDITOR ../ffwd16.obj ../ffwd17.obj
