from lab2_robotics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the link of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # Code almost identical to the one from lab2_robotics...
    assert link <= len(revolute), "Link value exceeds the number of revolute joints" 
    J_full = jacobian(T, revolute)
    J_link = np.copy(J_full)
    J_link[:, link:] = 0
    return J_link
    

def compute_quaternion(R):
    # Extract the elements of the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    
    # Compute the quaternion elements
    epsilon = np.zeros(3)
    epsilon[0] = 0.5 * np.where((r32 - r23) >= 0, 1, -1) * np.sqrt(np.maximum(0, r11 - r22 - r33 + 1))
    epsilon[1] = 0.5 * np.where((r13 - r31) >= 0, 1, -1) * np.sqrt(np.maximum(0, r22 - r33 - r11 + 1))
    epsilon[2] = 0.5 * np.where((r21 - r12) >= 0, 1, -1) * np.sqrt(np.maximum(0, r33 - r11 - r22 + 1))
    
    w  = 0.5 * np.sqrt(np.trace(R) + 1)
    return w, epsilon.reshape(3,1)

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof
    
    def getLinkTransform(self, link): # David
        '''
        Method to get the transformation for a selected link.
        
        Argument:
        selected link (integer): index of the link

        Returns:
        (double): Link transformation
        '''
        return self.T[link]

    def getLinkJacobian(self, link):
        '''
        Method to get the Jacobian for a selected link.
        '''
        # Llamar a getLinkTransform and then to jacobianLink
        return jacobianLink(self.T, self.revolute, link)
    
'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name        # task title
        self.sigma_d = desired  # desired sigma
        self.FFVel = None       # Attribute to be defined in the subclasses
        self.K = None           # Attribute to be defined in the subclasses
        self.active = 1         # Will be updated in the subclasses
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err

    def setFFVel(self, value):
        ''' 
        Method setting the feedforward velocity.

        Arguments:
        value(Numpy array): value of the feedforward velocity
        '''
        self.FFVel = value

    def getFFVel(self):
        '''
        Method returning the feedforward velocity.
        '''
        return self.FFVel
    
    def setK(self, value):
        ''' 
        Method setting the gain matrix K.

        Arguments:
        value(Numpy array): value of the gain matrix K
        '''
        self.K = value
    
    def getK(self):
        '''
        Method returning the gain matrix K.
        '''
        return self.K
    



'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, FFVel=np.zeros((2,1)), K=np.eye(2), link=None):
        super().__init__(name, desired)
        self.J = np.zeros((2,3))    # Initialize with proper dimensions
        self.err = np.zeros((2,1))  # Initialize with proper dimensions
        self.setFFVel(FFVel)        # Initialize with proper dimensions
        self.setK(K)                # Initialize with proper dimensions
        self.link = link
        
    def update(self, robot):
        dof = robot.getDOF()
        if self.link is None:
            self.J = robot.getEEJacobian()[:2,:] # Update task Jacobian
            self.err = self.getFFVel() + self.getK() @ (self.getDesired() - robot.getEETransform()[:2,3].reshape(2,1)) # Update task error
        else:
            self.J = robot.getLinkJacobian(self.link)[:2,:dof]     # Change the value to properly select the column of the joint
            self.err = self.getFFVel() + self.getK() @ (self.getDesired() - robot.getLinkTransform(self.link)[:2,3].reshape(2,1))
            pass
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, FFVel=0, K=1, link=None):
        super().__init__(name, desired)
        self.J = np.zeros((1,3))    # Initialize with proper dimensions
        self.err = 0                # Initialize with proper dimensions
        self.setFFVel(FFVel)
        self.setK(K)
        self.link = link

    def update(self, robot):
        if self.link is None:
            self.J = robot.getEEJacobian()[-1,:].reshape(1,3)  # Update task Jacobian
            T = robot.getEETransform()
        else:
            self.J = robot.getLinkJacobian(self.link)[-1,:].reshape(1,3)
            T = robot.getLinkTransform(self.link)
        
        cos_theta_n = T[0,0]
        sin_theta_n = T[1,0]

        orie = np.arctan2(sin_theta_n, cos_theta_n)
        self.err = self.getFFVel() + self.getK() * (self.getDesired() - orie)

        # Previous Implementation
        # R_B = robot.getEETransform()[0:3,0:3]
        # R_d = robot.T[self.getDesired()][0:3,0:3]
        # w,e = compute_quaternion(R_B)
        # wd,ed = compute_quaternion(R_d)
        # self.err = (w*ed - wd*e - np.cross(e.squeeze(), ed.squeeze()).reshape(-1,1))[:2,:] # Update task error

'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, FFVel=np.zeros((3,1)), K=np.eye(3), link=None):
        super().__init__(name, desired)
        self.J = np.zeros((3,3))    # Initialize with proper dimensions
        self.err = np.zeros((3,1))  # Initialize with proper dimensions
        self.setFFVel(FFVel)
        self.setK(K)
        self.link = link
        
    def update(self, robot):
        p2D_desired = self.getDesired()[:2].reshape(2,1)    # Retrieve from input array desired pose for Position2D task
        p2D_FFVel = self.getFFVel()[:2,:]   # Retrieve feedfoward velocity for Position2D task
        p2D_K = self.getK()[:2,:2]          # Retrieve matrix gain for Position2D task
        task_pose2D = Position2D("End-effector position", p2D_desired, p2D_FFVel, p2D_K, link=self.link)

        o2D_desired = self.getDesired()[2]      # Retrieve from input array desired orientation
        o2D_FFVel = self.getFFVel()[-1,:][0]    # Retrieve feedfoward velocity. [0] to pick the value from array
        o2D_K = self.getK()[-1,-1]              # Retrieve matrix gain for orientation task
        task_orie2D = Orientation2D("End-effector orientation", o2D_desired, o2D_FFVel, o2D_K, link=self.link)
        
        # Update both tasks
        task_pose2D.update(robot)
        task_orie2D.update(robot)

        self.J = np.vstack((task_pose2D.J, task_orie2D.J))  # Update task Jacobian
        self.err = np.vstack((task_pose2D.err, task_orie2D.err))  # Update task error
        pass # to remove

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, joint, FFVel=0, K=1):
        super().__init__(name, desired)
        self.J = np.zeros((1,3))    # Initialize with proper dimensions
        self.err = 0                # Initialize with proper dimensions
        self.joint = joint - 1      # Selected joint 
        self.setFFVel(FFVel)
        self.setK(K)
        
    def update(self, robot):
        self.J[0,self.joint] = 1     # Update task Jacobian
        self.err = self.getFFVel() + self.getK() * (self.getDesired() - robot.getJointPos(self.joint)) # Update task error


class Obstacle2D(Task):
    '''
    Subclass of Task. Inequality tasks: Obstacle avoidance task
    '''
    def __init__(self, name, desired, thresholds, FFVel=0, K=1):
        super().__init__(name, desired)
        self.J = np.zeros((2,3))    # Initialization of task Jacobian
        self.err = 0                # Initialization of err
        self.thresholds = thresholds    # Act. and deact. thresholds
        self.setFFVel(FFVel)            # FeedFoward Vel
        self.setK(K)                # Gain
    
    def update(self, robot):
        self.J = robot.getEEJacobian()[:2,:3]   # Get top two row, to compute linear vel of EE
        self.err = self.getFFVel() + self.getK() * (np.linalg.norm(robot.getEETransform()[:2,3].reshape(2,1) - self.getDesired()))
        self.active = self.isActive(self.thresholds)    # Update active flag

    def isActive(self, thresholds):
        '''
        Method returns bool that says if a task is active or not
        '''
        err = self.getError()       # Error between obstacle pose and EE pose
        act_thr   = thresholds[0]   # Activation threshold
        deact_thr = thresholds[1]   # Deactivation threshold
        if self.active == 1 and err >= deact_thr:
            return 0
        elif self.active == 0 and err <= act_thr:
            return 1
        elif err <= deact_thr and self.active:
            return 1
        else:
            return 0

class JointLimit(Task):
    '''
    Subclass of Task. Inequality tasks: Joint limit task
    '''
    def __init__(self, name, desired, joint, qmin, qmax, thresholds):
        super().__init__(name, desired)
        self.J = np.zeros((1,3))    # Initialize with proper dimensions
        self.err = 0
        self.joint = joint - 1      # Selected joint 
        self.qmin = qmin
        self.qmax = qmax
        self.thresholds = thresholds

    def update(self, robot):
        self.J[0,self.joint] = 1     # Update task Jacobian
        self.err = 1                 # Error
        self.active = self.isActive(self.thresholds, robot) # Update active flag

    def isActive(self, thresholds, robot):
        '''
        Method returns bool that says if a task is active or not
        '''
        joint_pose = robot.getJointPos(self.joint)      # Get Joint Position
        act_thr   = thresholds[0]     # Activation threshold
        deact_thr = thresholds[1]     # Deactivation threshold
        
        if self.active == 0 and joint_pose >= (self.qmax - act_thr):
            return -1
        elif self.active == 0 and joint_pose <= (self.qmin + act_thr):
            return 1
        elif self.active == -1 and joint_pose <= (self.qmax - deact_thr):
            return 0
        elif self.active == 1 and joint_pose >= (self.qmin + deact_thr):
            return 0
        else:
            return self.active
