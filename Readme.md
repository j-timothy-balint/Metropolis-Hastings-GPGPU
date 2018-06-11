# Readme for GPGPU algorithm
This GPGPU algorithm written in CUDA was used as a part of a WPF C# application that was implemented to be used in the user study of the thesis "Narrative world creation with assisted environment filling" by Marijn Goedegebure. The thesis itself can be found here: https://repository.tudelft.nl/islandora/object/uuid%3A819b3f0e-bcac-4501-86aa-f23199fee8e1?collection=education. The C# application can be found here: https://bitbucket.org/marijngoedegebure/narrativeworldcreator/src

This GPGPU algorithm was implemented based on the research "Interactive Furniture Layout Using Interior Design Guidelines" and implementation by Paul Merrel et al.

This repository has three branches. The test-version branch was used for the user study version of the C# application and works with that master branch. The Parallel_Met-Hastings and master branch include more optimizations as the test-version suffered from some performance issues for larger sets of objects. The optimizations were implemented by T. Balint during his time as a Post Doc during his time at the TU Delft.

If you have any questions regarding the implementation you can contact me at marijngoedegebure @ gmail.com.

This GPGPU algorithm and the C# application are both available to the public under the MIT licence, see below.

----------------------------------------------------

Copyright 2018 Marijn Goedegebure

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.