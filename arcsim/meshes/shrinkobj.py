#!/usr/bin/python
# Usage: python shrinkobj.py <filename.obj> <shrinkfactor> <movedown> > outfile
# This script takes a .obj file and shrinks all the vertices by the specified
#   amount, and then shifts it down on the y-axis by the specified amount.
import sys

if(len(sys.argv) == 1):
    print("Usage: python shrinkobj.py <filename.obj> <shrinkfactor> <movedown> > outfile");
    sys.exit(1);
filename = sys.argv[1]

shrinkfactor = sys.argv[2] if len(sys.argv) >= 3 else 30
movedown = sys.argv[3] if len(sys.argv) >= 4 else .5

fd = open( filename )
content = fd.readline()
while (content != "" ):
    parts = content.split(" ");
    if parts[0] != 'v':
        print content;
        content = fd.readline().strip()
        continue
    for i in range(1,4):
        parts[i] = float(parts[i]) / shrinkfactor
    parts[3] = parts[3] - movedown
    print parts[0], parts[1], parts[2], parts[3];
    content = fd.readline()
