




class Phase_Based_TrajOpt():
    """Gait and Trajectory Optimization for Legged Systems Through Phase-Based End-Effector Parameterization
            Winkler et al.
        
        Pseudo Code:
            Quasi-Newton L-BFGS algo for NLP
            
            
        Formulation: (Collocation Method)
            r(t): Linear Center of Mass (CoM) position
            \theta (t): Orientation of CoM
            p_i(t): feet motion
            f_i(t): contact force for each foot
            \Delta T_{i,j}: Appropriate gait pattern

            6D base motion: 
                represented by linear CoM position r(t) \in \mathbb{R}^3
                and orientation Euler Angles \theta(t) \in \mathbb{R}^3
                (Fourth order polynomials of fixed durations strung together to create continuous spline, optimize over coefficients)
            Each foot's Motion:
                p_i(t) \in \mathbb{R}^3
                (Use multiple 3rd order polynomials per swing phase, and constant val \in \mathbb{R}^3 for stance phase)
            Each foot's force profile:
                f_i(t) \in \mathbb{R}^3
                multiple polynomials represent each stance phase and zero force is set during swing-phase
                Duration of each phase, and duration of each foot's polynomial is changed based on the optimized phase durations
                \Delta T_{i,j} \in \mathbb{R}^3
    """
    def __init__(self):
        pass















































