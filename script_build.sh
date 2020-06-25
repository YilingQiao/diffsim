sudo ./change_gcc.sh 4.8
cd arcsim/dependencies/
make 
cd ../..
sudo ./change_gcc.sh 5
make -j 8
cd pysim
ln -s ../arcsim/conf ./conf
ln -s ../arcsim/materials ./materials
ln -s ../arcsim/meshes ./meshes
cd ..
