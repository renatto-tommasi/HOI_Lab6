from lab5_robotics import *

class MobileManipulator:
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
        self.revoluteExt = [True, False] + revolute  # List of joint types extended with base joints
        self.r =  0.05          # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt) # Number of DOF of the system
        self.q = np.zeros((len(self.revolute),1)) # Vector of joint positions (manipulator)
        self.eta = np.zeros((3,1)) # Vector of base pose (position & orientation)
        self.update(np.zeros((self.dof,1)), 0.0) # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    def update(self, dQ, dt):
        # Update manipulator
        self.q += dQ[2:, 0].reshape(-1,1) * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]

        # Update mobile base pose
        delta = dQ[:2, 0].reshape(-1,1) * dt
        self.eta = self.eta + np.array([-np.cos(self.eta[2,0]), 0, np.sin(self.eta[2,0]), 0,0,1]).reshape((3,2)) @ delta





        # Base kinematics
        Tb = np.array([[np.cos(self.eta[2,0]),-np.sin(self.eta[2,0]),0, self.eta[0,0]+self.r*np.cos(self.eta[2,0])],
                       [np.sin(self.eta[2,0]), np.cos(self.eta[2,0]),0, self.eta[1,0]+self.r*np.sin(self.eta[2,0])],
                       [0,0,1,0],
                       [0,0,0,1]])           # Transformation of the mobile base


        # Combined system kinematics (DH parameters extended with base DOF)
        dExt = np.concatenate([np.array([0,0]), self.d])
        thetaExt = np.concatenate([np.array([self.eta[2,0],0]), self.theta])
        aExt = np.concatenate([np.array([0,np.linalg.norm(self.eta[:2,0])]), self.a])
        alphaExt = np.concatenate([np.array([0,0]), self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

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
        return self.q[joint-2]


    def getBasePose(self):
        return self.eta

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    ###
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]
    
# Rewriting to 
    
def kinematics(d, theta, a, alpha, Tb):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis
        Tb (Numpy Array)  Transformation of the mobile base

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [Tb] # Base transformation

    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    DoF = len(d)    # Number of Degrees of Freedom
    A = []          # Initialize List
    for i in range(DoF):
        Ai = DH(d[i], theta[i], a[i], alpha[i]) # Compute transformation matrix (i-1)T(i)
        A.append(Ai)
        if len(A) == 1:       # Handle first iteration
            Ti = Ai
            T.append(Ti)
        else: 
            Ti = T[-1] @ Ai   # Example. (0)T(2) = (0)T(1) x (1)T(2)
            T.append(Ti)
    return T



if __name__ == "__main__":
    d = np.zeros(3)                         # displacement along Z-axis
    theta = np.array([0.25, 0.5, 0.75])     # rotation around Z-axis
    alpha = np.zeros(3)                     # rotation around X-axis
    a = np.array([0.75, 0.5, 0.25])         # displacement along X-axis
    revolute = [True, True, True]                      # flags specifying the type of joints
    robot = MobileManipulator(d, theta, a, alpha, revolute)
