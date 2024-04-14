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
        #   JointLimit("Joint Limit", desired=None, joint=3, qmin=-np.pi/8, qmax=np.pi/8, thresholds=np.array([np.pi/36, np.pi/72])),
        #   Position2D("End-effector position", np.array([-0.5, 1.5]).reshape(2,1))
            Configuration2D("End-effector Configuration", np.array([-1, -1.0, 0]))
        ] 

# Retrieve Position2D index from tasks list
pose2d_idx = None
for idx, task in enumerate(tasks):
    if isinstance(task, Position2D) or isinstance(task, Configuration2D):    # Check if the task is of type Position2D
        pose2d_idx = idx          # Assign the index of the Position2D task
        break

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# My initialization
errors = {} # To store errors
i = 1 # Counter to know how many loops of the simulation we did
vel_evo = None # Array to store the evolution of velocity output
eta_evo = {}  # Dic to store mobile base position
EE_evo = {}   # Dic to store end-effector position
priority_flag = False

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-3,3))
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

# Plot Error
def plot_error(errors, time, tasks):
    for key, value in errors.items():
        plt.plot(time, value, label=f'({key})')
    
    # Find if there is a JointLimit task
    for task in tasks:
        if isinstance(task, JointLimit):
            # Draw dotted lines
            qmin = task.qmin
            qmax = task.qmax
            plt.axhline(y=qmin, color='r', linestyle='--')
            plt.axhline(y=qmax, color='r', linestyle='--')

    # Labels and legend
    plt.xlabel('Time[s]')
    plt.ylabel('Value[1]')
    plt.title(f'Task-Priority control ({len(errors)} tasks)')
    plt.legend()
    plt.grid()
    plt.xlim(left=time[0])

    # Display the plot
    plt.show()

# Plot evolution of velocity output
def vel_evolution(vectors, time):
    # Plot all vectors against time
    plt.plot(time, vectors[0,:], label='Velocity joint 1')
    plt.plot(time, vectors[1,:], label='Velocity joint 2')
    plt.plot(time, vectors[2,:], label='Velocity joint 3')
    plt.plot(time, vectors[3,:], label='Velocity joint 4')
    plt.plot(time, vectors[4,:], label='Velocity joint 5')

    # Add labels and legend
    plt.xlabel('Time[s]')
    plt.ylabel('Value[1]')
    plt.title('Evolution of Velocity Output from TP Algorithm')
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

def pose_evolution(eta, EE):
    for key, value in eta.items():
        plt.plot(value[0,:], value[1,:], label=f'({key}), Base Position')

    for key, value in EE.items():
        plt.plot(value[0,:], value[1,:], label=f'({key}), EE Position')

    # Labels and legend
    plt.xlabel('Time[s]')
    plt.ylabel('Value[1]')
    plt.title(f'Task-Priority control ({len(errors)} tasks)')
    plt.legend()
    plt.grid()
    # plt.xlim(left=time[0])

    # Display the plot
    plt.show()

# Store Error
def store_error(task, errors):
    # Identify dimension of error
    if isinstance(task, JointPosition):
        err = abs(task.getError().item())
    elif isinstance(task, JointLimit):
        err = robot.getJointPos(task.joint)
    elif isinstance(task, Configuration2D):
        err_p = np.linalg.norm(task.getError()[:2, 0])
        err_o = task.getError()[2, 0]
    else:
        err = np.linalg.norm(task.getError())

    # Store error in correct dictionary
    if isinstance(task, Configuration2D):
        if task.name + "_position" not in errors:
            errors[task.name + "_position"] = np.array([err_p])
        else:
            errors[task.name + "_position"] = np.concatenate((errors[task.name + "_position"], np.array([err_p])))

        if task.name + "_orientation" not in errors:
            errors[task.name + "_orientation"] = np.array([err_o])
        else:
            errors[task.name + "_orientation"] = np.concatenate((errors[task.name + "_orientation"], np.array([err_o])))
    else:
        if task.name not in errors:
            # Handle first iteration
            errors[task.name] = np.array([err])
        else:
            # Concatenate error based on the current task
            errors[task.name] = np.concatenate((errors[task.name], np.array([err])))
    return errors

def store_vel(dq, vel_evo):
    if vel_evo is None:
        vel_evo = dq
    else:
        vel_evo = np.concatenate((vel_evo, dq), axis=1)
    return vel_evo

def store_pose(robot, eta_evo, EE_evo):
    eta = robot.getBasePose()[:2,0].reshape(2,1)
    EE = robot.getEETransform()[:2,3].reshape(2,1)
    
    if robot.priority not in eta_evo:
        # Handle first iteration
        eta_evo[robot.priority] = eta
    else:
        # Concatenate error based on the current task
        eta_evo[robot.priority] = np.concatenate((eta_evo[robot.priority], eta), axis=1)

    if robot.priority not in EE_evo:
        # Handle first iteration
        EE_evo[robot.priority] = EE
    else:
        # Concatenate error based on the current task
        EE_evo[robot.priority] = np.concatenate((EE_evo[robot.priority], EE), axis=1)
    
    return eta_evo, EE_evo

POINTS = [[-1,-1],
          [1.5,-1],
          [1.5,0],
          [-1,0]]
COUNTER = 0

# Simulation initialization
def init():
    global tasks
    global pose2d_idx   # Stores idx of task Position2D
    global i, POINTS, COUNTER

    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    range = (-2, 2)     # Range for x and y to generate random point
    if i != 1:              # If we are not in the first loop enter
        if isinstance(tasks[pose2d_idx], Configuration2D):  # If task is Configuration2D class, change only the desired target of the end-effector position
            orie_d = tasks[pose2d_idx].getDesired()[2]      # Get the desired target for orientation part
            # tasks[pose2d_idx].setDesired(np.array([np.random.uniform(*range), np.random.uniform(*range), orie_d]))  # Generate random point inside range and set it as new desired 
            newPoint = POINTS[COUNTER].copy()
            newPoint.append(orie_d)
            newConfig = np.array(newPoint)
            tasks[pose2d_idx].setDesired(newConfig)
            COUNTER += 1
            if COUNTER == 4:
                COUNTER = 0
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
    global vel_evo, eta_evo, EE_evo
    global priority_flag
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    
    print("t: ", round(t, 2), "  |  ",i)
    if t == 9.9:
        i = i + 1

    if i == 3 and robot.priority == "R":
        print("T")
        robot.priority = "T"
    elif i == 4 and robot.priority == "T":
        print("RT")
        robot.priority = "RT"
    elif i == 5 and robot.priority == "RT":
        print("R")
        robot.priority = "R"
    
    
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
        eta_evo, EE_evo = store_pose(robot, eta_evo, EE_evo)
        errors = store_error(task, errors)

        err_x=task.getDesired()[0]-robot.getBasePose()[0]
        err_y=task.getDesired()[1]-robot.getBasePose()[1]
        err_yaw=robot.getBasePose()[2]-np.arctan2(err_y,err_x)
        distance=np.linalg.norm(err_x-err_y)

        err_yaw = wrap_angle(err_yaw)

        rot_for = [True, False]         # Rotate First, Forward First
        if distance > 3 and not rot_for[0] and rot_for[1]:
            print("Going To Goal")
            dq[:2] = move_to_goal([True, False], distance, err_yaw[0])
            break 
        elif distance > 1:
            dq[:2] = move_to_goal(rot_for, distance, err_yaw[0])
            break

        if task.active != 0:
            # print("Doing Task")
            Ji_bar = task.getJacobian() @ Pi_1  # Compute augmented Jacobian
            
            # Inverse Jacobians (DLS and pseudoinverse)

            W = np.diag([1, 1, 0.3, 0.3, 0.3])

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

    vel_evo = store_vel(dq, vel_evo)

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[pose2d_idx].getDesired()[0], tasks[pose2d_idx].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, tt, 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

new_tt = tt
for j in range(1,i):
    temp = np.arange(new_tt[-1], 10 * (j+1), dt)
    new_tt = np.concatenate((new_tt,temp))

size_errors = list(errors.values())[0].shape[0]     # Get the size of the first element of the dictionary no matter what key does it have
size_vel_evo = vel_evo.shape[1]
plot_error(errors, new_tt[0:size_errors], tasks)
vel_evolution(vel_evo, new_tt[0:size_vel_evo])
pose_evolution(eta_evo, EE_evo)