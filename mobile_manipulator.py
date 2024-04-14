from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d = np.zeros(3)                         # displacement along Z-axis
theta = np.array([0.25, 0.5, 0.75])     # rotation around Z-axis
alpha = np.zeros(3)                     # rotation around X-axis
a = np.array([0.75, 0.5, 0.25])         # displacement along X-axis
revolute = [True, True, True]                      # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task definition

tasks = [ 
          # Position2D("End-effector position", np.array([-0.5, 1.5]).reshape(2,1))
          Configuration2D("End-effector Configuration", np.array([-0.5,1.5,0,1,1,1]).reshape(6,1))
        ] 

# Retrieve Position2D index from tasks list
pose2d_idx = None
for idx, task in enumerate(tasks):
    if isinstance(task, Position2D) or isinstance(task, Configuration2D):    # Check if the task is of type Position2D
        pose2d_idx = idx          # Assign the index of the Position2D task
        break

# Simulation params
dt = 1.0/60.0

# My initialization
errors = {} # To store errors
i = 1 # Counter to know how many loops of the simulation we did

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks
    global pose2d_idx   # Stores idx of task Position2D
    global i

    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    range = (-1.0, 1.0)     # Range for x and y to generate random point
    if i != 1:              # If we are not in the first loop enter
        if isinstance(tasks[pose2d_idx], Configuration2D):  # If task is Configuration2D class, change only the desired target of the end-effector position
            # orie_d = tasks[pose2d_idx].getDesired()[2]      # Get the desired target for orientation part
            tasks[pose2d_idx].setDesired(np.array([np.random.uniform(*range), np.random.uniform(*range),0, np.random.uniform(2*np.pi), np.random.uniform(2*np.pi),np.random.uniform(2*np.pi)]))  # Generate random point inside range and set it as new desired 
        else:
            # If task is Position2D change desired
            tasks[pose2d_idx].setDesired(np.array([[np.random.uniform(*range)], [np.random.uniform(*range)]]))  # Random desired end-effector position
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global errors, i
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    
    print("t: ", round(t, 2))
    if t == 9.9:
        i = i + 1

    # Initialize null-space projector
    Pi_1 = np.eye(robot.getDOF())

    # Initialize output vector (joint velocity)
    dqi_1 = np.zeros((robot.getDOF(),1))

    # Loop over tasks
    Pi = Pi_1
    dq = dqi_1
    
    for task in tasks:
        # Update task state
        task.update(robot)  
        err_x=task.getDesired()[0]-robot.getBasePose()[0]
        err_y=task.getDesired()[1]-robot.getBasePose()[1]
        err_yaw=robot.getBasePose()[2]-np.arctan2(err_y,err_x)
        distance=np.linalg.norm(err_x-err_y)

        err_yaw = wrap_angle(err_yaw)

        rot_for = [True, False]         # Rotate First, Forward First
        if distance > 1.5:
            print("Going To Goal")
            dq[:2] = move_to_goal(rot_for, distance, np.abs(err_yaw[0]))
            break 



        # Identify dimension of error
        if isinstance(task,JointPosition) is True:
            err = abs(task.getError().item())
        elif isinstance(task,JointLimit) is True:
            err = robot.getJointPos(task.joint)
        else:
            err = np.linalg.norm(task.getError())

        # Store error in correct dictionary
        if task.name not in errors:
            # Handle first iteration
            errors[task.name] = np.array([err])
        else:
            # Concatenate error based on the current task 
            errors[task.name] = np.concatenate((errors[task.name], np.array([err])))

        if task.active != 0:
            print("Doing Task")
            Ji_bar = task.getJacobian() @ Pi_1  # Compute augmented Jacobian
            
            # Inverse Jacobians (DLS and pseudoinverse)

            W = np.diag([1, 0.01, 1, 1, 1])
            print(task.getError()[2])

            Ji_bar_DLS = DLS(Ji_bar, 0.1, W)   # W=np.array([[1,0,0], [0,1,0], [0,0,1]])
        
            Ji_bar_pinv = np.linalg.pinv(Ji_bar)

            dq = dqi_1 + (Ji_bar_DLS @ (task.active * task.getError() - (task.getJacobian() @ dqi_1)))  # Accumulate velocity
            Pi = Pi_1 - (Ji_bar_pinv @ Ji_bar) # Update null-space projector
        else:
            dq = dqi_1
            Pi = Pi_1

        Pi_1 = Pi
        dqi_1 = dq
    # ###
    #     print(f"Robot Pose: {robot.getBasePose()}")
    #     print(f"Desired: {task.getDesired()}")
    #     print(f"Error: {task.getError()}")

    #     err_x=task.getDesired()[0]-robot.getBasePose()[0]
    #     err_y=task.getDesired()[1]-robot.getBasePose()[1]
    #     err_yaw=robot.getBasePose()[2]-np.arctan2(err_y,err_x)

    #     if err_yaw > 0.001:
    #         dq[:2] = np.array([1,0]).reshape(-1,1)
    #         print(f"Heading Error: {err_yaw}")

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()