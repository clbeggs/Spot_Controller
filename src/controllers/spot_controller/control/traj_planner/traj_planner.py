"""Uses infinite horizon Linear Quadratic Regulator(LQR) to find necessary horizontal force to meet
    desired speed and direction.

    Given model, the IPC is constructed so that the CoM of pendulum and position
    ot the cart match the CoM and CoP of the character calculated using rest pose
    or average pose of reference motion.

    Args:
        desired speed:
        desired direction:
        character model
        environmental constraints
    Returns:
        Center of Mass (CoM)
        Center of Pressure (CoP)
        Pendulum Orientation Trajectories
"""

""" 
Pseudo Code:
    Input: Desired speed/direction
    goal_speed, goal_direction = input()

    # Scale down desired speed for the pendulum by cos(terr_theta)
    terr_theta = #average slope of terrain ### Computation in Appendix B
    goal_speed *= cos(terr_theta)

    cart_traj = LQR(goal_speed)



"""

""" 
References: 
    Momentum-mapped Inverted Pendulum Models for Controlling Dynamic Human Motions by Hodgins and Kwon
    Control Systems for Human Running using an Inverted Pendulum Model and a Reference Motion Capture Sequence by Kwon, Hodgins
"""

class IPC_Model():
    def __init__(self):
        pass
    

class Traj_Planner():
    def __init__(self):
        pass
    
    def get_COM_velocity(self):
        pass

    def get_pendulum_trajectory(self):
        pass
        #v = self.get_COM_velocity()

    def get_pendulum_direction(self):
        pass

    
