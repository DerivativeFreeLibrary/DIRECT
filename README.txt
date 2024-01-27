-----------------------------------------------------------
 How to use the derivative-free optimizer DIRECT for 
 bound constrained optimization problems
-----------------------------------------------------------
 The package provides a C version of the code.

0- Gunzip and untar the archive in a folder on your computer by
   issuing in a directory of your choice (ex. curdir) the command

   $> tar -zxvf DIRECT.tar.gz

2- Edit file curdir/problem_c.c to define your own objective function.
   In particular, modify the subroutines 
   setdim    : which sets problem dimension
   setbounds : which sets upper and lower bounds on the variables
   funct     : which defines the objective function

   For the Julia interface, edit file 'example.jl' to define:
   your own objective function in julia
   the number of variables (n)
   upper and lower bounds on the variables (ub, lb)
   the maximum number of intervals (maxint)
   the estimated global minimum value (fglob)

2- At command prompt in curdir execute 

     $> make
 
   which will create the executable 'direct' and the library 'libdirect.a'

4- execute

     $> ./direct

5- execution within Julia. From within the curdir directory let Julia start.
   At the Julia prompt type

     julia> include("example.jl")

