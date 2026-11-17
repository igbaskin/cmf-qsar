# A structure of ten thousand atoms on screen: what it costs, and whether it is the same picture

`python benchmarks/run_large_view.py --force`, written 2026-11-17 15:23:43.

## The normals of a mesh: pyqtgraph's loop against array operations

A mesh is shaded from one normal per vertex. pyqtgraph computes them in a Python loop over the vertices before the first frame; `instanced.vertex_normals` computes the same sum with array operations. The spheres are 10 x 18 meshes, as the view draws up to 2,000 atoms. The processor was 27 % busy with all programs when the stage began; each time is the fastest of 3 repeats.

| Mesh | Vertices | pyqtgraph's loop | Array operations | Largest angle between the two normals |
| --- | ---: | ---: | ---: | ---: |
| 256 spheres | 41,984 | 275 ms | 15 ms | 2.8e-02 degrees |
| 2048 spheres | 335,872 | 2.50 s | 135 ms | 2.8e-02 degrees |
| 4000 spheres | 656,000 | 4.69 s | 312 ms | 2.8e-02 degrees |
| isosurface of a 923-atom particle | 78,360 | 539 ms | 23 ms | 3.1e-02 degrees |

Everything below was measured on **Intel(R) RaptorLake-S Mobile Graphics Controller** (OpenGL 4.6.0 - Build 32.0.101.7088) in a view of 900 x 700 pixels, with the processor 21 % busy with all programs when the stage began. A machine without a screen keeps this measurement.

## Atoms as spheres: one mesh against copies of one sphere

The time to build the item in Python, the first frame (which includes computing the normals and sending the data to the graphics card), the median of 20 later frames while the view turns, and the size of the data sent to the graphics card, counted from the arrays (4 bytes a number). Each time is the fastest of 3 repeats, each from a new item. 'One mesh' is the drawing before plan item NT8, kept in the script as a copy of the old code.

| Structure | Drawing | Building the item | First frame | Later frames | Data sent to the graphics card |
| --- | --- | ---: | ---: | ---: | ---: |
| Pt icosahedron of 923 atoms | one mesh (before NT8) | 50 ms | 6.20 s | 3.0 ms | 9.6 MB |
| Pt icosahedron of 923 atoms | copies of one sphere | 3 ms | 4 ms | 3.5 ms | 0.1 MB |
| Pt icosahedron of 10179 atoms | one mesh (before NT8) | 616 ms | 67.37 s | 7.3 ms | 106.4 MB |
| Pt icosahedron of 10179 atoms | copies of one sphere | 12 ms | 11 ms | 10.1 ms | 0.7 MB |
| Pt icosahedron of 49049 atoms | copies of one sphere | 93 ms | 23 ms | 16.8 ms | 3.1 MB |
| graphene sheet, 9800 atoms, 14491 bonds | one mesh (before NT8) | 11.86 s | 31.62 s | 2.8 ms | 24.1 MB |
| graphene sheet, 9800 atoms, 14491 bonds | copies of one stick | 53 ms | 5 ms | 3.7 ms | 1.9 MB |
| graphene sheet, 9800 atoms, 14491 bonds | lines | 5 ms | 4 ms | 2.4 ms | 0.8 MB |

## Is it the same picture

300 spheres of different sizes and colours and 60 sticks, drawn the old way, as copies, and by the fallback the new item uses on a card without instanced drawing (it builds the merged mesh itself). A pixel is drawn when it differs from the background in either picture.

| Pictures compared | Pixels drawn | Pixels that differ by more than 8 of 255 levels | Largest difference (of 255) | Pixels drawn in one picture only |
| --- | ---: | ---: | ---: | ---: |
| old / instanced | 269,498 | 2 | 166 | 0 |
| old / fallback | 269,498 | 3 | 170 | 0 |
| instanced / fallback | 269,498 | 1 | 170 | 0 |

At most 3 of the 269,498 drawn pixels differ by more than 8 levels, and no pixel is drawn in one picture and not in the other: every sphere and stick covers the same pixels. Of the 6 differing pixels of the three comparisons, 6 have no differing neighbour. A single pixel that differs while its neighbours agree is where two surfaces are at nearly the same depth: the rounding of the vertex positions (on the graphics card for the copies, in Python for a mesh) then decides which one is in front.

## In the windows

The time from the change of a control to the first frame of the crystal window, which includes everything the window computes for the view.

| What is shown | Atoms in the cell | To the first frame | Later frames | Pixels drawn |
| --- | ---: | ---: | ---: | ---: |
| crystal window, a Pt particle of 10179 atoms as balls | 10,179 | 108 ms | 12.7 ms | 327,299 |
| crystal window, cod_1545083.cif (256 atoms) with its images to 14.9 A as dots | 256 | 218 ms | 4.9 ms | 445,844 |
| crystal window, cod_1545083.cif (256 atoms) with its images to 14.9 A as balls | 256 | 195 ms | 10.8 ms | 1,817,669 |

Two views in one application, each drawing 50 spheres and a triangle, as the main window and the crystal window do when both are open:

| OpenGL contexts of the views | Pixels drawn by the first view | By the second | Drawing errors |
| --- | ---: | ---: | ---: |
| shared (the application now) | 28,835 | 28,835 | 0 |
| separate (before) | 28,835 | 0 | 2 |
