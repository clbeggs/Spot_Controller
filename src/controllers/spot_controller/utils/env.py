

""" 
Randomize given env. files.

This is all ugly and hacky code, but it's only called once at the start so it's ok I guess.
"""

import sys
import os
import numpy as np
import random

class WebotsEnv():
    def __init__(self):
        print("randomenv")

    def random_env(self):
        # TODO: Change start and goal position
        env_types = ['rock_env', 'shape_env', 'platform_env']
        env_types = ['rock_env'] #TODO: Add new world files
        random.shuffle(env_types)
        
        num_rocks = 70
        center = -10.15
        radius = (0, 3, 3)
        
        random_world = env_types[0]
        
        filename = "./../../worlds/" + random_world + ".wbt"
        
        file_obj = open(filename, 'r')
        file_contents = file_obj.read()
        file_obj.close()
        
        filename = "./../../worlds/tmp.wbt"
        
        with open(filename, 'r+') as tmp_file:
            tmp_file.truncate(0)
            tmp_file.write(file_contents)
        
        with open(filename, 'a') as tmp_file:



            # Add Obstacles
            obstacles = []
            for i in range(num_rocks):
                rock_x = np.random.uniform(-3,3)
                rock_z = np.random.uniform(-3, 3) + center
                obstacles.append([rock_x, 0, rock_z])

                rock = ("Rock17cm {\n",
                        "  translation %f 0 %f\n" % (rock_x, rock_z),
                        '  name "rock %d"\n' % (i),
                        "  physics Physics {\n",
                        "    density -1\n",
                        "    mass 2\n",
                        "  }\n",
                        "}\n")
                tmp_file.write(''.join(rock))

    

 





















