from lab5_robotics import *

deg90 = np.pi / 2


class MobileManipulator:
    """
    Constructor.

    Arguments:
    d (Numpy array): list of displacements along Z-axis
    theta (Numpy array): list of rotations around Z-axis
    a (Numpy array): list of displacements along X-axis
    alpha (Numpy array): list of rotations around X-axis
    revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    """

    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute

        # List of joint types extended with base joints
        self.revoluteExt = [True, False] + self.revolute

        self.r = 0  # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt)  # Number of DOF of the system

        # Vector of joint positions (manipulator)
        self.q = np.zeros((len(self.revolute), 1))

        # Vector of base pose (position & orientation)
        self.eta = np.zeros((3, 1))

        # Initialise robot state
        self.update(np.zeros((self.dof, 1)), 0.0)

    """
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    """

    def update(self, dQ, dt):
        # Update manipulator
        self.q += dQ[2:, 0].reshape(-1, 1) * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]

        # Update mobile base pose
        forward_vel = dQ[1, 0]
        angular_vel = dQ[0, 0]
        yaw = self.eta[2, 0]

        self.eta += dt * np.array(
            [forward_vel * np.cos(yaw), forward_vel * np.sin(yaw), angular_vel]
        ).reshape(3, 1)

        # Base kinematics
        x, y, yaw = self.eta.flatten()
        # Base kinematics
        Tb = translation(x, y) @ rotation_z(yaw)          # Transformation of the mobile base
  # Transformation of the mobile base

        ### Additional rotations performed, to align the axis:
        # Rotate Z +90 (using the theta of the first base joint)
        # Rotate X +90 (using the alpha of the first base joint)
        ## Z now aligns with the forward velocity of the base
        # Rotate X -90 (using the alpha of the second base joint)
        ## Z is now back to vertical position
        # Rotate Z -90 (using the theta of the first manipulator joint)

        # Modify the theta of the base joint, to account for an additional Z rotation
        self.theta[0] -= deg90

        # Combined system kinematics (DH parameters extended with base DOF)
        dExt = np.concatenate([np.array([0, self.r]), self.d])
        thetaExt = np.concatenate([np.array([deg90, 0]), self.theta])
        aExt = np.concatenate([np.array([0, 0]), self.a])
        alphaExt = np.concatenate([np.array([deg90, -deg90]), self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

    """ 
        Method that returns the characteristic points of the robot.
    """

    def drawing(self):
        return robotPoints2D(self.T)

    """
        Method that returns the end-effector Jacobian.
    """

    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

    """
        Method that returns the end-effector transformation.
    """

    def getEETransform(self):
        return self.T[-1]

    """
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    """

    def getJointPos(self, joint):
        return self.q[joint - 2]

    def getBasePose(self):
        return self.eta

    """
        Method that returns number of DOF of the manipulator.
    """

    def getDOF(self):
        return self.dof

    ###
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]
    


def translation(x,y):
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,0],
                     [0,0,0,1]])

def rotation_z(yaw):
    return np.array([[np.cos(yaw),-np.sin(yaw),0,0],
                     [np.sin(yaw),np.cos(yaw),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )


def move_to_goal(rot_for, distance, err_yaw):
    v = 0
    w = 0
    Kv = 0.5
    Kw = 0.8
    if rot_for[0] and err_yaw > 0.1:
        w = Kw * err_yaw


    if rot_for[1] or err_yaw < 0.1:
        v = Kv * distance

    return np.array([w ,v]).reshape(-1,1)
             