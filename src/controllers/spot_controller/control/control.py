




class Contact_Invariant_Optim():
    def __init__(self, joints, cameras, leds):
        self.joints = joints
        self.cameras = cameras
        self.leds = leds
    
    def get_nearest_point(self):
        """Reference: Last paragraph of section 4.1 in paper"""
        pass

    """Discovery of Complex Behaviors through Contact-Invariant Optimization by Mordatch et. al
    
        'discovering suitable contact sets is the central goal of optimization in our approach'
    
        Formulation:
            s: real valued solution vector, encodes movement trajectory and auxiliary variables
                Trajectory is represented by function approximators (splines) 

            t: time, 1 \leq t \leq T, where T is the overall movement time

            q_t(s): Character pose, must be well-defined function of s at each discrete point in time t
    
            c_{i, \phi(t)} (s) \geq 0: auxiliary variables, included in s
            
            T: overall movement time, partitioned into K intervals/phases, such that 1 \leq \phi(t) \leq K is the
                index of the phase to which each time step t belongs

            K: Number of intervals/phases, predefined such that durations are equal (can also be optimized)

            1 \leq i \leq N: index over "end effectors", where an "end effector" is a specific surface patch on one of the
                rigid bodies, the only places where contact forces can be exerted.

            p_i (q) \in \mathbb{R}^3: function that returns center of patch i
            
        The CIO method computes optimal solution s^* by minimizing composite obj. function L(s):
            L(s) = L_{CI}(s) + L_{Physics}(s) + L_{Task} (s) + L_{Hint} (s)
        where:
            L_{CI} is the contact-invariant cost introduced in the paper
            L_{Physics} penalizes physical violations, (soft cost, not hard constraint)
            L_{Task} specifies task objectives
            L_{Hint} optional, can be used to provide hints such as ZMP-like costs

        L_{CI} COST FUNCTION ---------
            L_{CI} is defined to be:
                L_{CI} (s) = \sum \limits _ {i} c_{i, \phi(t)} (s) \left( ||e_{i,t} (s)||^2 + || \dot{e}_{i,t} (s) ||^2 \right)
            where e_{i,t} is a 4D contact violation vector for end-effector i at time t,
                encodes the misalignment in both position and orientation.

            and if c_{i, \phi(t)} is a large value, it means that the end effector i should be in contact with the env. during 
                the entire movement phase \phi(t) to which time t belongs.

            The first 3 components of e are the difference vector between end effector position p_i(q_t)
                and the nearest point on any surface in the environment(including body segments)
            
            The last component of e is the angle betwee the surface normal at the nearest point and the surface normal 
                at the end effector

            L_{CI} penalizes both e and its velocity \dot{e} which corresponds to slip

            "nearest point" uses a soft-min instead of min. n_j(p) denotes the actual nearest point to p on surface j.
            N_j(p) = \frac{1}{1 + ||p - n_j(p)||^2 k}

            where k = 10^4 is a smoothness param
            A virtual nearest pt is obtained by normalizing weights of N_j to sum to 1, and computing
                weighted avg of n_j's.
            This makes it so if p_i is far from any surface, (c_i is large) the cost L_{CI} will push it towards some average
                of the surface "mass". When p_i gets close to a surface, it will be pusehd towards the nearest 
                pt. on that surface. THIS makes L_{CI} smooth.
        
        L_{Physics} COST FUNCTION ---------
            This cost has two components:
                1.) General, depends on contact dynamics 
                2.) Specific to simplified model of multi-body dynamics,
                    can be replaced w/ full physics model w/o modifying rest of method (described below)

            f \in \mathbb{R}^{6N}: vector of contact forces acting on all N end effectors.
                    6D vector because method allows torsion around the surface normal, and the origin
                    of the contact force is allowed to move inside the end effector surface patch









    """
