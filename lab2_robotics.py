import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    # 2. Multiply matrices in the correct order (result in T).
    T_di = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,d],
                     [0,0,0,1]])
    
    T_thetai = np.array([[np.cos(theta), -np.sin(theta),0,0],
                         [np.sin(theta), np.cos(theta),0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
    
    T_ai = np.array([[1,0,0,a],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]])
    
    T_alphai = np.array([[1,0,0,0],
                         [0,np.cos(alpha),-np.sin(alpha),0],
                         [0,np.sin(alpha),np.cos(alpha),0],
                         [0,0,0,1]])

    T = T_di @ T_thetai @ T_ai @ T_alphai
    return T

def kinematics(d, theta, a, alpha, Tb=np.eye(4)):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [Tb] # Base transformation

    for tz, rz, tx, rx in zip(d, theta, a, alpha):
        matrix = DH(tz, rz, tx, rx)
        T_from_base = T[-1]@matrix
        T.append(T_from_base)

    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    
    # T = T[1:]   # Remove base transformation (1st element)
    Tn = T[-1]    # Save last transformation matrix, in this case (0)T(2)
    pn = Tn[:-1,3].reshape(3,1)     # Save origin of frame n (end effector)

    J = None    # Initialize as None to handle 1st iteration later
    for i, Ti in enumerate(T):
        if np.array_equal(Ti, T[-1]): # We don't compute column for the last T (0)T(n)
            return J
        zi_1 = Ti[:-1,2].reshape(3,1) # Get z-axis 
        pi_1 = Ti[:-1,3].reshape(3,1) # Get frame origin
        if revolute[i] == True:
            # Angular Joint
            Ji_top = np.cross(zi_1[:,0], (pn - pi_1)[:,0]).reshape(3,1) # We slice them (a[:,0]) to convert them to 1D arrays because numpy.cross() requires 1D arrays.
            Ji_bottom = zi_1

        elif revolute[i] == False:
            # Prismatic Joint
            Ji_top = zi_1
            Ji_bottom = np.zeros_like(zi_1)
        
        if isinstance(J, np.ndarray):
            temp0 = np.vstack((Ji_top,Ji_bottom))
            J = np.hstack((J,temp0))
        else:   # Case J = None (1st iteration)
            J = np.vstack((Ji_top,Ji_bottom))

# Damped Least-Squares
def DLS(A, damping, W=None):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    if W is None:
        W = np.eye(A.shape[1])
    W_inv = np.linalg.inv(W)
    A_inv = W_inv @ A.T @ np.linalg.inv(A @ W_inv @ A.T + damping**2 * np.eye(A.shape[0]))
    return A_inv # Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P