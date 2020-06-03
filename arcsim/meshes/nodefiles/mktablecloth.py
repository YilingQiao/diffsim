from math import sqrt
numVerts = 24*4 + 1
print numVerts, "2 0 0"
print "0 0 0"
for i in xrange(1, 25):
    num=sqrt(i/25.0);
    print (i-1)*4+1,  num,  num
    print (i-1)*4+2, -num,  num
    print (i-1)*4+3, -num, -num
    print (i-1)*4+4,  num, -num
