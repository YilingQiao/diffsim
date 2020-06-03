**********************************************************************
      ARCSim v0.2.1: Adaptive Refining and Coarsening Simulator

      Rahul Narain, Armin Samii, Tobias Pfaff, and James O'Brien
              {narain,samii,tpfaff,job}@cs.berkeley.edu
**********************************************************************

This software is provided freely for non-commercial use. See the
LICENSE file for details. If you use the software in your research,
please cite the following papers:

Rahul Narain, Armin Samii, and James F. O'Brien. "Adaptive Anisotropic
Remeshing for Cloth Simulation". ACM Transactions on Graphics,
31(6):147:1-10, November 2012. Proceedings of ACM SIGGRAPH Asia 2012,
Singapore.

Rahul Narain, Tobias Pfaff, and James F. O'Brien. "Folding and
Crumpling Adaptive Sheets". ACM Transactions on Graphics,
32(4):51:1-8, July 2013. Proceedings of ACM SIGGRAPH 2013, Anaheim.

======================================================================
 Usage
======================================================================

For compilation instructions, see the INSTALL file. If you're on a
Mac, make sure you have the latest gcc, as described there.

The simulator has various modes, which are called by running it with
the format

    bin/arcsim <command> [<args>]

The various command and their inputs can be discovered by running the
executable with no arguments. The main simulation-related commands are
as follows.

* simulate <scene-file> [<out-dir>]

    Reads the simulation description given in <scene-file> (try
    conf/sphere.json, then see "Scene files" below) and runs the
    simulation while displaying the results in an OpenGL window. Press
    Space to run the simulation once the window opens, or 'S' to
    advance one timestep at a time. If <out-dir> is given, the results
    of the simulation are saved in that directory. The format of the
    output is described in "Output files".

* simulateoffline <scene-file> [<out-dir>]

    Same as 'simulate' but without the OpenGL display. (And you don't
    have to press Space to start the simulation.) Useful for running
    on a remote server over SSH.

* resume <out-dir> <resume-frame>

    Resumes an incomplete simulation saved in <out-dir>, starting at
    the given frame number. (Note that if the original scene file has
    been changed since the simulation was first run, those changes
    will not be respected in the resumed simulation.)

* resumeoffline <out-dir> <resume-frame>

    Same as 'resume' but without the OpenGL display.

* replay <out-dir> [<sshot-dir>]

    Replays the results of the simulation stored in <out-dir>. The
    left and right arrow keys step through the frames; Shift+arrow and
    Alt+arrow jump 10 and 100 frames. Hitting Space plays back all the
    frames as fast as possible. The Home key returns to frame 0. If
    <sshot-dir> is given, the rendered images are also saved there
    upon playback.

----------------------------------------------------------------------
 Using the OpenGL interface
----------------------------------------------------------------------

Space:          start simulation/playback
Left drag:      rotate
Middle drag:    translate
Scroll wheel:   scale
Esc:            exit

----------------------------------------------------------------------
 Scene files
----------------------------------------------------------------------

The scene configuration files are written in JSON syntax
(http://www.json.org/). A number of example scenes are provided in the
conf/ directory. The sphere interaction example, conf/sphere.json, is
a good "hello world" scene to check that the simulator is working.

A scene file consists of a single JSON object containing many
name/value pairs describing the parameters and contents of the
simulation. For a full description of all the options, see the file
conf/sphere-annotated.json, which is a heavily commented version of
conf/sphere.json and describes all the settings that can be used.

======================================================================
 Creating meshes for simulation
======================================================================

Simulation meshes are saved as OBJ files, with the parametrization
space embedding stored in texture coordinates, and other simulation
data (node velocities, plasticity data, etc.) in custom fields. These
meshes can be imported or viewed as usual in 3D software.

If you want to use a flat sheet of material cut in an arbitrary shape,
as specified by a triangle mesh in the xy-plane, this requires no
extra effort. The simulator will simply assume the rest shape to be
identical to the initial input. For this to work correctly, however,
(i) the input mesh must have no texture coordinates specified; (ii)
the mesh should lie in the xy-plane (its initial pose in the scene can
be changed using the scene file); and (iii) all faces should be
positively oriented, i.e. their vertices must be listed in
counter-clockwise order.

Simulating a garment composed of multiple panels sewn together
requires a little initial setting-up. At present, we only support
garments that form manifolds with consistent orientation. Having more
than two faces meeting along an edge, as occurs with pockets, or
joining the back side of one panel with the front side of another,
will break the simulation.

First, we model the panels laid flat and positively oriented in the
xy-plane as above, and save them in a nonoverlapping arrangement in a
single OBJ file. Call this the parametrization-space mesh. If two
boundaries are to be joined into a seam, care must be taken in this
step to have them discretized consistently, with both sides having the
same number of vertices and equal lengths of corresponding edges.

Second, we take the same mesh and move its vertices into the desired
initial positions in world space, say around a posed character. We
indicate seams and darts by placing two or more boundary vertices very
close to each other; in the next step, such vertices will be fused
together. Here, though, we only move vertices, but must not modify the
mesh topology. This is saved as the world-space mesh.

Finally, the parametrization-space and world-space meshes are combined
to form a mesh in which the seam vertices have been fused, but the
intrinsic flat shapes of the panels are still intact in texture
coordinates. This is done with the command

    bin/arcsim merge <param-mesh> <world-mesh> <out-mesh> [<thresh>]

where the optional <thresh> (default 0.01) sets the maximum distance
threshold between vertices to be merged.

For an example of this procedure, see the meshes named tshirtm.obj,
tshirtw.obj, and tshirt.obj in the meshes/ direectory.

======================================================================
 Output files
======================================================================

The output of the simulation consists of a number of files:

* conf.json is a copy of the scene file that was used to run the
  simulation. This is useful for resuming, replaying, and general
  future reference.

* <frame>_<i>.obj is the mesh for the <i>th cloth at the given frame
  number. A simulation may have multiple articles of cloth, e.g. the
  pants, t-shirt, and vest of the kicking character.

* obs_<j>.obj is a copy of the geometry of the <j>th obstacle in its
  reference pose.

* <frame>obs<j>.txt is an XML fragment describing the transformation
  of the <j>th obstacle at the given frame. It is only used to plug
  into the Mitsuba scene file when rendering. The resume and replay
  code ignores it, instead reconstructing the obstacle motion from the
  specification in conf.json.

* timing is a dump of the wall-clock compute time spent per time step
  in each routine of the simulation.
