













# Fast and Flexible Multilegged Locomotion Using Learned Centroidal Dynamics
Uses inverted pendulum model.








## TODO
Finish installing Paradiso, shared object files are:
libpardiso600-GNU800-X86-64.so (More recent, use this one)
libpardiso600-GNU720-X86-64.so

Have to install HSL for Ipopt, they will email with license, check for that.


## Dependencies:
Ipopt
   Intel Math Kernel Library 
    HSL Mathematical Software Library
    Paradiso shared object file in /usr/local/lib/
    iterative Linear Quadratic Optimal Control - https://github.com/ethz-adrl/control-toolbox

## Compiling (w/ Paradiso):
https://pardiso-project.org/manual/manual.pdf
For clangd, add `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` to `cmake`


## Parameterization of Optimization Quantities: 
### Kinematic Model
 - Approximates each foot's possible movement by cube of length 2b centered @ nominal position of each foot p̅ᵢ
    relative to the Center of Mass, so joint limits are not violated if every foot is in the cube.
    Cube is given by: 
    pᵢ(t) ∈ ℛᵢ(r,θ) ⇔ | R(θ) [pᵢ(t) − r(t)] − p̅ᵢ | < b, where R(θ) is the rotation matrix from the world to base frame.

### Dynamic Model
Uses simplified Centroidal dynamics.

Center of Mass linear acceleration r̈ is given by: 
mr̈(t) = ∑ⁿ fᵢ(t) − mg

Center of Mass angular acceleration ω̇ is given by:
Iω̇(t) + ω(t) × Iω(t) = ∑ⁿ fᵢ(t) × ((r(t) − pᵢ(t))


Where 
- m is the mass of the robot
- nᵢ is the number of feet
- g is gravity
- ω(t) is angular velocity which is calculated from optimized Euler angles θ(t) and rates θ̇(t)
- Constant rotational inertia I I ∈ ℝ³ × ³ (assumes limb masses are negligible compared to torso or limbs do not deviate significantly from default pose)
The assumptions above make it so that the dynamics of the robot is independent from joint configuration.











## Reference: 
Gait and Trajectory Optimization for Legged Systems Through Phase-Based End-Effector Parameterization

