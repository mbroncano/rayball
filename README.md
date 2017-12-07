Rayball
=======

Small, simple Whitted-style Cornell box ray tracer, written in C++

![rayball](/rayball.png)

## Features
* Realtime performance
* Parallel render using OpenMP
* Portable OpenGL/GLUT display
* Distributed raytracer with soft shadows and antialiasing
* Primitives: spheres, triangles and quads
* Materials: diffuse (Lambert and Phong illumination), specular, transparent/refractive (implements Fresnell's equations)
* Texture mapping (PPM format) with Bilinear filter
* Disco party mode!

## Todo
* Scene file loader
* Acceleration structure (kd-trees or BHV)
