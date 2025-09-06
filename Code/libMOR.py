
import sys
sys.path.append('../../../../CppToPython')

import numpy as np
import GeDiM4Py as gedim

import torch
import torch.nn as nn

problemData = None
lib = None

dofs = None
strongs = None

stiffness    = None
forcingTermf = None
forcingTermg = None

stiffnessN    = None
forcingTermfN = None
forcingTermgN = None

B = None

device = None
ff = None

#############
### GeDiM ###
#############

def SetGeometry(aMax,L,order):
    global problemData, lib
    global dofs, strongs

    lib = gedim.ImportLibrary("../../../../CppToPython/release/GeDiM4Py.so")
    config = {'GeometricTolerance': 1.0e-8}
    gedim.Initialize(config,lib)

    # From this snippet «1» corresponds to the Dirichlet condition while «0», evidently, to the Neumann one
    # Actually from the lab «FullOrderModel», they are just arbitrary markers to identify boundary conditions
    # However, I don't know what «DiscretizationType» means
    domain = {'SquareEdge': L,
              'VerticesBoundaryCondition': [1,1,1,1],
              'EdgesBoundaryCondition': [1,1,1,1],
              'DiscretizationType': 1,
              'MeshCellsMaximumArea': aMax}
    [meshInfo, mesh] = gedim.CreateDomainSquare(domain,lib)

    discreteSpace = {'Order': order,
                     'Type': 1,
                     'BoundaryConditionsType': [1,2]}
    [problemData, dofs, strongs] = gedim.Discretize(discreteSpace,lib)
    # gedim.PlotDofs(mesh,dofs,strongs)

    return [dofs, strongs, mesh] # Returns the number of degrees of freedom

# These are the unitary [implicit] constants a and c in the weak equation
# «a*int_Ω[∇δu·∇v]+c*int_Ω[μ₀δue^{μ₁u_k}v]=-a*int_Ω[∇uₖ·∇v]-c*int_Ω[μ₀e^{μ₁u_k}v]+int_Ω[(g-μ₀/μ₁)v]»,
def Constant_a(numPoints,points):
    valuesa = np.ones(numPoints,order='F')
    return valuesa.ctypes.data

def Constant_c(numPoints,points):
    valuesc = np.ones(numPoints,order='F')
    return valuesc.ctypes.data
# For «order='F'» see https://numpy.org/doc/2.2/reference/generated/numpy.ones.html#:~:text=Whether%20to%20store%20multi%2Ddimensional%20data%20in%20row%2Dmajor%20(C%2Dstyle)%20or%20column%2Dmajor%20(Fortran%2Dstyle)%20order%20in%20memory in essence it's telling Python how to store the data in memory

# Creates two vector of length «numPoints» from «u_x» and «u_y», and combines them to evaluate the derivative of g
def AssignmentDerf(numPoints,points,u,u_x,u_y):
    vecu_x = gedim.make_nd_array(u_x,numPoints,np.double)
    vecu_y = gedim.make_nd_array(u_y,numPoints,np.double)
    valuesDf = np.zeros((2,numPoints),order='F')
    valuesDf[0,:] = vecu_x
    valuesDf[1,:] = vecu_y
    return valuesDf.ctypes.data

# Creates a unitary vector of length «numPoints»
def Ones(numPoints,points):
    valuesOne = np.ones(numPoints,order='F')
    return valuesOne.ctypes.data

# Creates a unitary matrix of length «2⨯numPoints»
def OnesDerivative(numPoints, points):
    valuesOneD = np.ones((2,numPoints),order='F')
    return valuesOneD.ctypes.data

# Creates a zero vector of length «numPoints»
def Zeros(numPoints,points):
    valuesZero = np.zeros(numPoints,order='F')
    return valuesZero.ctypes.data

# Creates a zero matrix of length «2⨯numPoints»
def ZerosDerivative(direction,numPoints,points):
    valuesZeroD = np.zeros(numPoints,order='F')
    return valuesZeroD.ctypes.data

##############
### Part 0 ###
##############

# Creates a vector of length «numPoints» for the values of f, as defined by «f=μ₀/μ₁»
def Assignmentf(numPoints,points): return Ones(numPoints,points) # m0/m1

def InnerProductMatrix():

    # Inner product
    [stiffness, stiffnessStrong] = gedim.AssembleStiffnessMatrix(Ones,problemData,lib)

    # Norm (‖u‖²_{H¹₀(Ω)}=‖u‖²_{(L²(Ω))}+‖∇u‖²_{(L²(Ω))^d})
    # [reaction, reactionStrong] = gedim.AssembleReactionMatrix(Domain,problemData,lib)

    # Semi-norm (‖u‖²_{H¹_{0,Γ_D}(Ω)}=‖∇u‖²_{(L²(Ω))^d})
    A = stiffness

    # This matrix is used to compute the inner product of the Hilbert space H¹_{0,Γ_D}(Ω), based on
    # the seminorm ∥∇v∥²_{(L²(Ω))^d}=int_Ω{∇v·∇v}; indeed assume v=Σ_{i=1}^{Nδ}[vᵢφᵢ] given a
    # basis {φᵢ}_{i=1}^{Nδ}, then it holds:

    # ∥∇v∥²_{(L²(Ω))^d}  =int_Ω{∇v·∇v}
    #                   =Σ_{i=1}^{Nδ}Σ_{j=1}^{Nδ}[vᵢint_Ω{∇φᵢ·∇φⱼ}vⱼ]
    #                   =Σ_{i,j=1}^{Nδ}[vᵢA_{i,j}vⱼ]
    #                   =vᵀAv

    # The stiffness [symmetric] matrix A in vᵀAv corresponds to «inner_product» and in this exercise
    # it's a sum for the stiffness matrix due to the domain subdivision: the first contribution comes
    # from Ω₂ while the second from Ω₁, and using the linearity of the integral one has

    # ∥∇v∥²_{(L²(Ω))^d} =int_Ω{∇v·∇v}
    #                   =int_Ω₁{∇v·∇v}+int_Ω₂{∇v·∇v}
    #                   =vᵀA₁v+vᵀA₂v
    #                   =vᵀ(A₁+A₂)v
    #                   =vᵀAv ⟹ A=A₁+A₂

    # Lastly the reason why the seminorm is considered, instead of the full norm, comes from the fact that if
    # there is at least a Dirichlet condition the former is equivalent to the latter; for more information
    # see p. 16 in the file «Numerical Methods for PDE {23-09-2024}» where if «Γ_D≠∅ [...] the Poincaré
    # inequality [...] holds true in V [and] we can endow V with the norm ∥v∥_V=∥∇v∥_{(L²(Ω))^d}»

    A_strong = stiffnessStrong

    return [A, A_strong]

# Creates a vector of length «numPoints» to evaluate the non linear part of int_Ω[μ₀δue^{μ₁u_k}v] as well as «-int_Ω[(μ₀/μ₁)e^{μ₁u_k}v]»
def AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1):
    vecu     = gedim.make_nd_array(u,numPoints,np.double)
    valuesNL = np.exp(m1*vecu)
    return valuesNL.ctypes.data

def AssembleAffineMatrix(g):
    global stiffness, forcingTermf, forcingTermg

    # Definition of the affine stiffness matrix
    [stiffness, _] = gedim.AssembleStiffnessMatrix(Constant_a,problemData,lib) # Matrix form of the weak integral «int_Ω[∇δu·∇v]»

    forcingTermf = gedim.AssembleForcingTerm(Assignmentf,problemData,lib) # Vector form of the weak integral «int_Ω[v]» later multiplied by «μ₀/μ₁»
    
    # Creates a vector of length «numPoints» for the values of f, as defined by «f=g+μ₀/μ₁=100sin(2πx₀)cos(2πx₁)+μ₀/μ₁»
    def Assignmentg(numPoints,points):
        matPoints = gedim.make_nd_matrix(points,(3,numPoints),np.double)
        valuesf   = g(matPoints)
        return valuesf.ctypes.data    

    # Definition of the affine reduced forcing functions
    forcingTermg = gedim.AssembleForcingTerm(Assignmentg,problemData,lib) # Vector form of the weak integral «int_Ω[gv]»

def FOMSol(L,A,order,m0,m1,g):

    [dofs, strongs, mesh] = SetGeometry(A,L,order)
    AssembleAffineMatrix(g)
    [u_k,u_strong] = FOMSolF(m0,m1)

    return [mesh, dofs, strongs, u_k, u_strong]

def FOMSolF(m0,m1):

    def AssignmentNonLinearm1(numPoints,points,u,u_x,u_y): return AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1)

    # Newton parameters
    resNorm = 1.0    # Residual norm
    solNorm = 1.0    # Solution norm
    NewTol  = 1.0e-6 # Newton tolerance
    maxI    = 100    # Maximum iterations
    numI    = 1      # Number of iterations

    # Initial solution
    u_k = np.zeros(problemData['NumberDOFs'],order='F')
    u_strong = np.zeros(problemData['NumberStrongs'],order='F')


    # Main cycle
    while numI < maxI and resNorm > NewTol * solNorm:

        [nonLinear, _]   = gedim.AssembleNonLinearReactionMatrix(Constant_c,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Matrix form of the weak integral «int_Ω[δue^{μ₁u_k}v]»
        
        forcingTermNL    = gedim.AssembleNonLinearForcingTerm(Ones,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Vector form of the weak «-int_Ω[e^{μ₁u_k}v]»
        
        forcingTermStiff = gedim.AssembleNonLinearDerivativeForcingTerm(OnesDerivative,AssignmentDerf,u_k,u_strong,problemData,lib) # Vector form of the weak integral «int_Ω[∇uₖ·∇v]»
        
        du = gedim.LUSolver(stiffness + m0*nonLinear, \
                            forcingTermg + (m0/m1)*forcingTermf \
                            - (m0/m1)*forcingTermNL - forcingTermStiff,
                            lib) # Vector step δu from the linear system

        u_k = u_k + du # Solution update
        
        resNorm = gedim.ComputeErrorL2(Zeros,du,np.zeros(problemData['NumberStrongs'],order='F'),lib)
        solNorm  = gedim.ComputeErrorL2(Zeros,u_k,u_strong,lib)

        # u_normH1  = gedim.ComputeErrorH1(ZerosDerivative,u_k,u_strong,lib)
        
        # solNorm = u_normL2;
        # resNorm = du_normL2;
        
        # print("dofs","h","normL2","normH1","residual","iteration","max_iteration")
        # print(problemData['NumberDOFs'],\
        #       '{:.2e}'.format(problemData['H']),\
        #       '{:.2e}'.format(u_normL2),\
        #       '{:.2e}'.format(u_normH1),\
        #       '{:.2e}'.format(residual_norm / u_normL2),\
        #       '{:d}'.format(num_iteration),\
        #       '{:d}'.format(max_iterations))
        # The code «'{:.16e}'.format()» and «'{:d}'.format()» are just way to convert a number into a string while formatting
        # the output as scientific with 16 digits after the decimal point and as integer in decimal base respectively
        
        numI += 1
    return [u_k,u_strong]

##############
### Part 1 ###
##############

# Forcing function definition
def g1(matPoints): return 100*np.sin(2*np.pi*matPoints[0,:])*np.cos(2*np.pi*matPoints[1,:])

# Creates a vector of length «numPoints» for the values of f, as defined by «g1=100sin(2πx₀)cos(2πx₁)»
def Assignmentg1(numPoints,points):
    matPoints = gedim.make_nd_matrix(points,(3,numPoints),np.double)
    valuesf   = g1(matPoints)
    return valuesf.ctypes.data


def AssembleAffineMatrix1():
    global stiffness, forcingTermf, forcingTermg

    # Definition of the affine stiffness matrix
    [stiffness, _] = gedim.AssembleStiffnessMatrix(Constant_a,problemData,lib) # Matrix form of the weak integral «int_Ω[∇δu·∇v]»

    # Definition of the affine forcing functions
    forcingTermf = gedim.AssembleForcingTerm(Assignmentf,problemData,lib) # Vector form of the weak integral «int_Ω[v]» later multiplied by «μ₀/μ₁»   
    forcingTermg = gedim.AssembleForcingTerm(Assignmentg1,problemData,lib) # Vector form of the weak integral «int_Ω[gv]»

# Similar to the previous definition but with fixed
# geometry, so to better compare the FOM and ROM times
def FOMSolF1(m0,m1):

    def AssignmentNonLinearm1(numPoints,points,u,u_x,u_y): return AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1)

    # Newton parameters
    resNorm = 1.0    # Residual norm
    solNorm = 1.0    # Solution norm
    Newtol  = 1.0e-6 # Newton tolerance
    maxI    = 100    # Maximum iterations
    numI    = 1      # Number of iterations

    # Initial solution
    u_k = np.zeros(problemData['NumberDOFs'],order='F')
    u_strong = np.zeros(problemData['NumberStrongs'],order='F')

    # Main loop
    while numI < maxI and resNorm > Newtol * solNorm:

        [nonLinear, _]   = gedim.AssembleNonLinearReactionMatrix(Constant_c,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Matrix form of the weak integral «int_Ω[δue^{μ₁u_k}v]»
        
        forcingTermNL    = gedim.AssembleNonLinearForcingTerm(Ones,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Vector form of the weak «-int_Ω[e^{μ₁u_k}v]»
        
        forcingTermStiff = gedim.AssembleNonLinearDerivativeForcingTerm(OnesDerivative,AssignmentDerf,u_k,u_strong,problemData,lib) # Vector form of the weak integral «int_Ω[∇uₖ·∇v]»
        
        du = gedim.LUSolver(stiffness + m0*nonLinear,
                            forcingTermg + (m0/m1)*forcingTermf
                            - (m0/m1)*forcingTermNL - forcingTermStiff,
                            lib) # Vector step δu from the linear system
        
        u_k = u_k + du # Solution update
        
        resNorm = gedim.ComputeErrorL2(Zeros,du,np.zeros(problemData['NumberStrongs'],order='F'),lib)
        solNorm  = gedim.ComputeErrorL2(Zeros,u_k,u_strong,lib)
        
        # u_normH1  = gedim.ComputeErrorH1(ZerosDerivative,u_k,u_strong,lib)
        
        # print("dofs","h","normL2","normH1","residual","iteration","max_iteration")
        # print(problemData['NumberDOFs'],\
        #       '{:.2e}'.format(problemData['H']),\
        #       '{:.2e}'.format(u_normL2),\
        #       '{:.2e}'.format(u_normH1),\
        #       '{:.2e}'.format(residual_norm / u_normL2),\
        #       '{:d}'.format(num_iteration),\
        #       '{:d}'.format(max_iterations))
        # The code «'{:.16e}'.format()» and «'{:d}'.format()» are just way to convert a number into a string while formatting
        # the output as scientific with 16 digits after the decimal point and as integer in decimal base respectively

        numI += 1

    return [u_k,u_strong]

def FOMSol1(L,A,order,m0,m1):

    [dofs, strongs, mesh] = SetGeometry(A,L,order)
    AssembleAffineMatrix1()
    [u, u_strong] = FOMSolF1(m0,m1)

    return [mesh, dofs, strongs, u, u_strong]


def AssembleReducedAffineMatrix1():
    global stiffnessN, forcingTermfN, forcingTermgN

    # Definition of the affine stiffness reduced matrix
    stiffnessN = np.transpose(B)@stiffness@B

    # Definition of the affine reduced forcing functions
    forcingTermfN = np.transpose(B)@forcingTermf
    forcingTermgN = np.transpose(B)@forcingTermg

# «ROMSol» doesn't have neither L nor A as «FOMSol» since
# a reduced solution is defined by fixing the geometry
def ROMSolF1(m0,m1):

    def AssignmentNonLinearm1(numPoints,points,u,u_x,u_y): return AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1)

    # Newton parameters
    resNorm = 1.0    # Residual norm
    solNorm = 1.0    # Solution norm
    Newtol  = 1.0e-6 # Newton tolerance
    maxI    = 100    # Maximum iterations
    numI    = 1      # Number of iterations

    # Initial solution
    u_k = np.zeros(problemData['NumberDOFs'],order='F')
    u_strong = np.zeros(problemData['NumberStrongs'],order='F')


    # Main cycle
    while numI < maxI and resNorm > Newtol * solNorm:

        [nonLinear, _]   = gedim.AssembleNonLinearReactionMatrix(Constant_c,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Matrix form of the weak integral «int_Ω[δue^{μ₁u_k}v]»
        nonLinearN = np.transpose(B)@nonLinear@B
        
        forcingTermNL    = gedim.AssembleNonLinearForcingTerm(Ones,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Vector form of the weak «-int_Ω[e^{μ₁u_k}v]»
        forcingTermNLN = np.transpose(B)@forcingTermNL

        forcingTermStiff = gedim.AssembleNonLinearDerivativeForcingTerm(OnesDerivative,AssignmentDerf,u_k,u_strong,problemData,lib) # Vector form of the weak integral «int_Ω[∇uₖ·∇v]»
        forcingTermStiffN = np.transpose(B)@forcingTermStiff

        # Vector step δu from the linear system
        duN = np.linalg.solve(stiffnessN + m0*nonLinearN,
                              forcingTermgN + (m0/m1)*forcingTermfN
                              - (m0/m1)*forcingTermNLN - forcingTermStiffN)
        # Alternatevely it's possible to use «gedim.LUSolver(reduced_rhs,reduced_lhs,lib)

        # Solution update
        du = B@duN
        u_k = u_k + du
        
        resNorm = gedim.ComputeErrorL2(Zeros,du,np.zeros(problemData['NumberStrongs'],order='F'),lib)
        solNorm  = gedim.ComputeErrorL2(Zeros,u_k,u_strong,lib)
        
        numI += 1

    return [u_k,u_strong]

##############
### Part 2 ###
##############

# Forcing function definition
def g2(matPoints,m0): return 100*np.sin(2*np.pi*m0*matPoints[0,:])*np.cos(2*np.pi*m0*matPoints[1,:])

# Creates a vector of length «numPoints» for the values of f, as defined by «g2=100sin(2πμ₀x₀)cos(2πμ₀x₁)»
def Assignmentg2(numPoints,points,m0):
    matPoints = gedim.make_nd_matrix(points,(3,numPoints),np.double)
    valuesf   = g2(matPoints,m0)
    return valuesf.ctypes.data


def AssembleAffineMatrix2():
    global stiffness, forcingTermf
    
    # Definition of the affine stiffness reduced matrix
    [stiffness, _]  = gedim.AssembleStiffnessMatrix(Constant_a,problemData,lib) # Matrix form of the weak integral «int_Ω[∇δu·∇v]»

    forcingTermf = gedim.AssembleForcingTerm(Assignmentf,problemData,lib) # Vector form of the weak integral «int_Ω[v]» later multiplied by «μ₀/μ₁»

# FOM solution with fixed geometry, so to better compare the FOM and ROM times
def FOMSolF2(m0,m1):

    def AssignmentNonLinearm1(numPoints,points,u,u_x,u_y): return AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1)

    def Assignmentg2m0(numPoints,points): return Assignmentg2(numPoints,points,m0)

    # Definition of the forcing function vector
    forcingTermg = gedim.AssembleForcingTerm(Assignmentg2m0,problemData,lib) # Vector form of the weak integral «int_Ω[gv]»

    # Newton parameters
    resNorm = 1.0    # Residual norm
    solNorm = 1.0    # Solution norm
    Newtol  = 1.0e-6 # Newton tolerance
    maxI    = 100    # Maximum iterations
    numI    = 1      # Number of iterations

    # Initial solution
    u_k = np.zeros(problemData['NumberDOFs'],order='F')
    u_strong = np.zeros(problemData['NumberStrongs'],order='F')


    # Main cycle
    while numI < maxI and resNorm > Newtol * solNorm:

        [nonLinear, _]   = gedim.AssembleNonLinearReactionMatrix(Constant_c,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Matrix form of the weak integral «int_Ω[δue^{μ₁u_k}v]»
        
        forcingTermNL    = gedim.AssembleNonLinearForcingTerm(Ones,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Vector form of the weak «-int_Ω[e^{μ₁u_k}v]»
        
        forcingTermStiff = gedim.AssembleNonLinearDerivativeForcingTerm(OnesDerivative,AssignmentDerf,u_k,u_strong,problemData,lib) # Vector form of the weak integral «int_Ω[∇uₖ·∇v]»
        
        du = gedim.LUSolver(stiffness + m0*nonLinear,
                            forcingTermg + (m0/m1)*forcingTermf
                            - (m0/m1)*forcingTermNL - forcingTermStiff,
                            lib) # Vector step δu from the linear system
        
        u_k = u_k + du # Solution update
        
        resNorm = gedim.ComputeErrorL2(Zeros,du,np.zeros(problemData['NumberStrongs'],order='F'),lib)
        solNorm  = gedim.ComputeErrorL2(Zeros,u_k,u_strong,lib)

        # u_normH1  = gedim.ComputeErrorH1(ZerosDerivative,u_k,u_strong,lib)
        
        # print("dofs","h","normL2","normH1","residual","iteration","max_iteration")
        # print(problemData['NumberDOFs'],\
        #       '{:.2e}'.format(problemData['H']),\
        #       '{:.2e}'.format(u_normL2),\
        #       '{:.2e}'.format(u_normH1),\
        #       '{:.2e}'.format(residual_norm / u_normL2),\
        #       '{:d}'.format(num_iteration),\
        #       '{:d}'.format(max_iterations))
        # The code «'{:.16e}'.format()» and «'{:d}'.format()» are just way to convert a number into a string while formatting
        # the output as scientific with 16 digits after the decimal point and as integer in decimal base respectively

        numI += 1

    return [u_k,u_strong]

def FOMSol2(L,A,order,m0,m1):

    [dofs, strongs, mesh] = SetGeometry(A,L,order)
    AssembleAffineMatrix2()
    [u, u_strong] = FOMSolF2(m0,m1)

    return [mesh, dofs, strongs, u, u_strong]


def AssembleReducedAffineMatrix2():
    global stiffnessN, forcingTermfN

    # Definition of the affine stiffness reduced matrix
    stiffnessN = np.transpose(B)@stiffness@B

    # Definition of the affine stiffness reduced vector
    forcingTermfN = np.transpose(B)@forcingTermf

# «ROMSol» doesn't have neither L nor A as «FOMSol» since
# a reduced solution is defined by fixing the geometry
def ROMSolF2(m0,m1):

    def AssignmentNonLinearm1(numPoints,points,u,u_x,u_y): return AssignmentNonLinear(numPoints,points,u,u_x,u_y,m1)

    def Assignmentg2m0(numPoints,points): return Assignmentg2(numPoints,points,m0)

    # Definition of the affine reduced forcing functions
    forcingTermg = gedim.AssembleForcingTerm(Assignmentg2m0,problemData,lib) # Vector form of the weak integral «int_Ω[gv]»
    forcingTermgN = np.transpose(B)@forcingTermg

    # Newton parameters
    resNorm = 1.0    # Residual norm
    solNorm = 1.0    # Solution norm
    Newtol  = 1.0e-6 # Newton tolerance
    maxI    = 100    # Maximum iterations
    numI    = 1      # Number of iterations

    # Initial solution
    u_k = np.zeros(problemData['NumberDOFs'],order='F')
    u_strong = np.zeros(problemData['NumberStrongs'],order='F')


    # Main cycle
    while numI < maxI and resNorm > Newtol * solNorm:

        [nonLinear, _]   = gedim.AssembleNonLinearReactionMatrix(Constant_c,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Matrix form of the weak integral «int_Ω[δue^{μ₁u_k}v]»
        nonLinearN = np.transpose(B)@nonLinear@B
        
        forcingTermNL    = gedim.AssembleNonLinearForcingTerm(Ones,AssignmentNonLinearm1,u_k,u_strong,problemData,lib) # Vector form of the weak «-int_Ω[e^{μ₁u_k}v]»
        forcingTermNLN = np.transpose(B)@forcingTermNL

        forcingTermStiff = gedim.AssembleNonLinearDerivativeForcingTerm(OnesDerivative,AssignmentDerf,u_k,u_strong,problemData,lib) # Vector form of the weak integral «int_Ω[∇uₖ·∇v]»
        forcingTermStiffN = np.transpose(B)@forcingTermStiff

        # Vector step δu from the linear system
        duN = np.linalg.solve(stiffnessN + m0*nonLinearN,
                              forcingTermgN + (m0/m1)*forcingTermfN
                              - (m0/m1)*forcingTermNLN - forcingTermStiffN)
        # Alternatevely it's possible to use «gedim.LUSolver(reduced_rhs,reduced_lhs,lib)

        # Solution update
        du = B@duN
        u_k = u_k + du
        
        resNorm = gedim.ComputeErrorL2(Zeros,du,np.zeros(problemData['NumberStrongs'],order='F'),lib)
        solNorm  = gedim.ComputeErrorL2(Zeros,u_k,u_strong,lib)
        
        numI += 1

    return [u_k,u_strong]


############
### PINN ###
############

def SharedPINNStructure():
    # The inputs are the space variables (x,y) and the parameters (μ₀,μ₁)
    nNI = 4  # Number of input nodes

    nHL = 5  # Number of hidden layers
    nNH = 50 # Number of hidden nodes per layer

    # The output is the predicted value of the PDE solution u(x,y;μ₀,μ₁)
    nNO = 1  # Number of output nodes

    return [nNI, nHL, nNH, nNO]

# Physics Informed Neural Network (PINN) class definition
class PINN(nn.Module):
    def __init__(self,nNI,nHL,nNH,nNO):
        super(PINN,self).__init__()
        self.il = nn.Linear(nNI,nNH) # Input layer
        self.hl = nn.ModuleList([nn.Linear(nNH, nNH) for _ in range(nHL)]) # Hidden layers
        self.ol = nn.Linear(nNH,nNO) # Output layer
        self.activation = torch.tanh

        # Before the for cycle every hidden layer had to be defined manually :(
        # self.hl1  = nn.Linear(nNH,nNH) # Hidden layer 1
        # self.hl2  = nn.Linear(nNH,nNH) # Hidden layer 2
        # self.hl3  = nn.Linear(nNH,nNH) # Hidden layer 3
        # self.hl4  = nn.Linear(nNH,nNH) # Hidden layer 4
        # self.hl5  = nn.Linear(nNH,nNH) # Hidden layer 5
        # self.hl6  = nn.Linear(nNH,nNH) # Hidden layer 6
        # self.hl7  = nn.Linear(nNH,nNH) # Hidden layer 7
        # self.hl8  = nn.Linear(nNH,nNH) # Hidden layer 8
        # self.hl9  = nn.Linear(nNH,nNH) # Hidden layer 9
        # self.hl10 = nn.Linear(nNH,nNH) # Hidden layer 10

    def forward(self,x,y,m0,m1):
        input  = torch.cat([x,y,m0,m1],axis=1) # Concatenate the inputs along the columns
        output = self.activation(self.il(input))
        for layer in self.hl:
            output = self.activation(layer(output))
        output = self.ol(output)
        return output
        # Remark: if the inputs («x», «y», «m0» and «m1» in this case)
        # are vectors of dimension N, then each of the N rows of «input»
        # is treated as a separate 4 dimensional data point by the NN
    
        # Before the for cycle every hidden layer had to be defined manually :(
        # l1o    = torch.tanh(self.il(input)) # Layer 1 output
        # l2o    = torch.tanh(self.hl1(l1o))  # Layer 2 output
        # l3o    = torch.tanh(self.hl2(l2o))  # Layer 3 output
        # l4o    = torch.tanh(self.hl3(l3o))  # Layer 4 output
        # l5o    = torch.tanh(self.hl4(l4o))  # Layer 5 output
        # l6o    = torch.tanh(self.hl3(l5o))  # Layer 6 output
        # l7o    = torch.tanh(self.hl4(l6o))  # Layer 7 output
        # l8o    = torch.tanh(self.hl3(l7o))  # Layer 8 output
        # l9o    = torch.tanh(self.hl4(l8o))  # Layer 9 output
        # l10o   = torch.tanh(self.hl4(l9o))  # Layer 10 output
        # l11o   = torch.tanh(self.hl4(l10o)) # Layer 11 output
        # output = self.ol(l11o)

    # A layer can be seen as the space between two columns of nodes,
    # thus the number of node columns will be that of the layers +1

def g1T(x,y,m0): return 100*torch.sin(2*torch.pi*x)*torch.cos(2*torch.pi*y)

def g2T(x,y,m0): return 100*torch.sin(2*torch.pi*m0*x)*torch.cos(2*torch.pi*m0*y)

def ResPDE(x,y,m0,m1,net):
    u = net(x,y,m0,m1)

    # Remember that «R=-∆u+(μ₀/μ₁)(e^{μ₁u}-1)-g», thus d²u/dx²,
    # d²u/dy², (μ₀/μ₁)(e^{μ₁u}-1) and g have to be defined
    ux  = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(),x,create_graph=True)[0]

    uy  = torch.autograd.grad(u.sum(),y,create_graph=True)[0]
    uyy = torch.autograd.grad(uy.sum(),y,create_graph=True)[0]

    # u = torch.clamp(u,min=-5,max=5)

    lu  = uxx + uyy  # Laplacian
    nl  = (m0/m1)*(torch.exp(m1*u) - 1) # Nonlinear term
    g   = ff(x,y,m0) # Forcing term

    # lu = torch.clamp(lu,min=-maxVal,max=maxVal)
    # nl = torch.clamp(nl,min=-maxVal,max=maxVal)

    maxVal = 2e2
    lhs = torch.clamp(-lu+nl,min=-maxVal,max=maxVal)
    rhs = g

    res = lhs - rhs # PDE residual

    # iMax = torch.argmax(res).item()
    # str = (f"{iMax}"
    #        f"\tx: {x[iMax].item():.4e}"
    #        f"\ty: {y[iMax].item():.4e}"
    #        f"\tm0: {m0[iMax].item():.4e}"
    #        f"\tm1: {m1[iMax].item():.4e}"
    #        f"\tlu: {lu[iMax].item():.4e}"
    #        f"\tnl: {nl[iMax].item():.4e}"
    #        f"\tg: {g[iMax].item():.4e}")
    # print(str)

    return res

def HomogenousTensor(nP,value):
    tensor = torch.full((nP,1),value,dtype=torch.float,device=device)
    return tensor

# Sampled points in each line indipendently
def RandomBP(nP,low,high):
    # Random boundary points
    x0BC = np.random.uniform(low,high,(nP,1)) # y=0
    # x0BC = Variable(torch.from_numpy(x0BC).float(),requires_grad=False)
    x0BC = torch.from_numpy(x0BC).float().to(device)
    x0BC.requires_grad = False

    x1BC = np.random.uniform(low,high,(nP,1)) # y=1
    # x1BC = Variable(torch.from_numpy(x1BC).float(),requires_grad=False)
    x1BC = torch.from_numpy(x1BC).float().to(device)
    x1BC.requires_grad = False

    y0BC = np.random.uniform(low,high,(nP,1)) # x=0
    # y0BC = Variable(torch.from_numpy(y0BC).float(),requires_grad=False)
    y0BC = torch.from_numpy(y0BC).float().to(device)
    y0BC.requires_grad = False

    y1BC = np.random.uniform(low,high,(nP,1)) # x=1
    # y1BC = Variable(torch.from_numpy(y1BC).float(),requires_grad=False)
    y1BC = torch.from_numpy(y1BC).float().to(device)
    y1BC.requires_grad = False

    return [x0BC,x1BC,y0BC,y1BC]

# Sampled points in the domain Ω
def RandomIP(nP,low,high):
    # Random internal points
    xIP = np.random.uniform(low,high,(nP,1)) # x-coordinates of random internal points
    # xIP = Variable(torch.from_numpy(xIP).float(),requires_grad=True)
    xIP = torch.from_numpy(xIP).float().to(device)
    xIP.requires_grad = True

    yIP = np.random.uniform(low,high,(nP,1)) # y-coordinates of random internal points
    # yIP = Variable(torch.from_numpy(yIP).float(),requires_grad=True)
    yIP = torch.from_numpy(yIP).float().to(device)
    yIP.requires_grad = True

    return [xIP,yIP]

def RandomParam(nP,low,high):
    # Random parameters
    m0 = np.random.uniform(low,high,(nP,1)) # μ₀ random values
    # m0 = Variable(torch.from_numpy(m0).float(),requires_grad=True)
    m0 = torch.from_numpy(m0).float().to(device)
    m0.requires_grad = True

    m1 = np.random.uniform(low,high,(nP,1)) # μ₁ random values
    # m1 = Variable(torch.from_numpy(m1).float(),requires_grad=True)
    m1 = torch.from_numpy(m1).float().to(device)
    m1.requires_grad = True

    return [m0,m1]

def lossBC(net,MSELoss,nBP,zeros,ones):

    # Random boundary parameters
    [m0,m1] = RandomParam(nBP,0.1,1)

    # BC Loss
    mseBC = []

    # Random boundary points
    [x0BC,x1BC,y0BC,y1BC] = RandomBP(nBP,0,1)

    # Bottom BC with y=0
    uBCP = net(x0BC,zeros,m0,m1) # Predicted PINN BC
    mseBC.append(MSELoss(uBCP,zeros)) # Bottom BC Loss

    # Upper BC with y=1
    uBCP = net(x1BC,ones,m0,m1) # Predicted PINN BC
    mseBC.append(MSELoss(uBCP,zeros)) # Upper BC Loss

    # Left BC with x=0
    uBCP = net(zeros,y0BC,m0,m1) # Predicted PINN BC
    mseBC.append(MSELoss(uBCP,zeros)) # Left BC Loss

    # Right BC with x=1
    uBCP = net(ones,y1BC,m0,m1) # Predicted PINN BC
    mseBC.append(MSELoss(uBCP,zeros)) # Right BC Loss

    mseBC = sum(mseBC)

    return mseBC

def lossRes(net,MSELoss,nIP,zeros):

    # Random internal parameters
    [m0,m1] = RandomParam(nIP,0.1,1)

    # Random internal points
    [xIP,yIP] = RandomIP(nIP,0,1)

    # PDE Loss
    resIP = ResPDE(xIP,yIP,m0,m1,net) # Internal point residual
    mseRes = MSELoss(resIP,zeros)

    return mseRes

def lossTot(mseBC,mseRes,l):

    # Balance of the two losses
    epsilon = 1e-8 # Small number to prevent division by zero
    alfa = 0.75

    # The terms r and t are considered so to have a smoother
    # transition of λ whenever either mseBC or mseRes suddenly change
    r = mseBC.item()/(mseRes.item() + epsilon)
    t = 1/(1 + r)
    l = alfa*l + (1-alfa)*t
    l = max(0.05,min(0.95,l))

    loss = l*mseBC + (1-l)*mseRes

    # The reasoning behind these multiplications for λ is to balance the two contribution.
    
    # Consider for instance the extreme case mseRes<<mseBC, then r->+∞ and t->0; this gives
    # l = alfa*l which means λ will tend to zero (0.05 in this case), i.e. l*mseBC≈0 while 
    # (1-l)*mseRes≈mseRes which balances the sum as now l*mseBC is similar to (1-l)mseRes

    # A similar reasoning is valid with  mseRes>>mseBC, so that, in essence, multiplying λ to
    # mseBC and 1-λ to mseRes tries to make the two order of magnitude of the addends equal


    # l = mseBC.item()/(mseRes.item() + epsilon)
    # loss = (1-l)*mseBC + l*mseRes

    return [loss, l]

class LoggerPINN:
    def __init__(self,optimiser,step):
        self.loss   = np.zeros((step,1))
        self.mseBC  = np.zeros((step,1))
        self.mseRes = np.zeros((step,1))
        self.l      = np.zeros((step,1))
        self.lr     = np.zeros((step,1))
        self.grad   = np.zeros((step,1))
        self.optimiser = optimiser
        self.last   = ''

    def log(self,e,step,mseBC,mseRes,loss,l,net):
        self.mseBC[e % step]  = mseBC.item()
        self.mseRes[e % step] = mseRes.item()
        self.loss[e % step]   = loss.item()
        self.l[e % step]      = l
        self.lr[e % step]     = self.optimiser.param_groups[0]['lr']
        
        norm = 0
        for p in net.parameters():
            if p.grad is not None:
                paramNorm = p.grad.data.norm(2)
                norm += paramNorm.item()**2
        self.grad[e % step] = norm**0.5

        if e % step == 0:
            meanBC   = np.mean(self.mseBC)
            meanRes  = np.mean(self.mseRes)
            meanLoss = np.mean(self.loss)
            meanl    = np.mean(self.l)
            meanlr   = np.mean(self.lr)
            meanGrad = np.mean(self.grad)

            self.last = (f"{e}"
                         f"\tmeanL: {meanLoss:.4e}"
                         f"\tmeanBC: {meanBC:.6e}"
                         f"\tmeanRes: {meanRes:.6e}"
                         f"\tlr: {meanlr:.0e}"
                         f"\tλ: {meanl:.2f}"
                         f"\tGrad: {meanGrad:.2f}")
            print(self.last) # flush=True,

            # sys.stdout.flush()
            # print(e, # flush=True,
            #       "\tmeanL:",   '{:.4e}'.format(meanLoss),
            #       "\tmeanBC:" , '{:.6e}'.format(meanBC),
            #       "\tmeanRes:", '{:.6e}'.format(meanRes),
            #       "\tlr:",      '{:.0e}'.format(meanlr),
            #       "\tλ:",       '{:.2f}'.format(meanl),
            #       "\tGrad:",    '{:.2f}'.format(meanGrad))

def buildSolPINN(net,m0,m1):

    n_dofs = dofs.shape[1]

    m0dofs = torch.full((n_dofs,1),m0,dtype=torch.float,device=device)
    m1dofs = torch.full((n_dofs,1),m1,dtype=torch.float,device=device)

    u_dofs = np.zeros((n_dofs,))
    x = torch.from_numpy(dofs[0,:].reshape(-1,1)).float().to(device)
    y = torch.from_numpy(dofs[1,:].reshape(-1,1)).float().to(device)
            
    if device == 'cpu':
        u_dofs = net(x,y,m0dofs,m1dofs).detach().numpy().reshape(-1,)
    else:
        u_dofs = net(x,y,m0dofs,m1dofs).cpu().detach().numpy().reshape(-1,)


    n_strong = strongs.shape[1]

    m0strong = torch.full((n_strong,1),m0,dtype=torch.float,device=device)
    m1strong = torch.full((n_strong,1),m1,dtype=torch.float,device=device)

    u_strong = np.zeros((n_strong,))
    x = torch.from_numpy(strongs[0,:].reshape(-1,1)).float().to(device)
    y = torch.from_numpy(strongs[1,:].reshape(-1,1)).float().to(device)

    if device == 'cpu':
        u_strong = net(x,y,m0strong,m1strong).detach().numpy().reshape(-1,)
    else:
        u_strong = net(x,y,m0strong,m1strong).cpu().detach().numpy().reshape(-1,)

    return [u_dofs, u_strong]

    # Reference
    # x = np.concatenate((dofs[0,:],strongs[0,:]),axis=0)
    # y = np.concatenate((dofs[1,:],strongs[1,:]),axis=0)
    # z = np.concatenate((u_k,u_strong), axis=0)


##############
### POD-NN ###
##############

def SharedPODNNStructure(N):
    # The inputs are the two parameters (μ₀,μ₁)
    nNI = 2  # Number of input nodes

    nHL = 5  # Number of hidden layers
    nNH = 50 # Number of hidden nodes per layer

    # The outputs are the components of the reduced space basis
    nNO = N  # Number of output nodes

    return [nNI, nHL, nNH, nNO]

class PODNN(nn.Module):

    def __init__(self,nNI,nHL,nNH,nNO):
        super(PODNN,self).__init__()
        self.il = nn.Linear(nNI,nNH)
        self.hl = nn.ModuleList([nn.Linear(nNH, nNH) for _ in range(nHL)]) # Hidden layers
        self.ol = nn.Linear(nNH,nNO)
        self.tanh = nn.Tanh()

    def forward(self,m0,m1):
        x = torch.cat([m0,m1],axis=1)
        x = self.tanh(self.il(x))
        for layer in self.hl:
            x = self.tanh(layer(x))
        x = self.ol(x)
        return x
    
    # A layer can be seen as the space between two columns of nodes,
    # thus the number of node columns will be that of the layers +1

class LoggerPODNN:
    def __init__(self,optimiser,step):
        self.loss   = np.zeros((step,1))
        self.lr     = np.zeros((step,1))
        self.grad   = np.zeros((step,1))
        self.optimiser = optimiser
        self.last   = ''

    def log(self,e,step,loss,net):
        self.loss[e % step]   = loss.item()
        self.lr[e % step]     = self.optimiser.param_groups[0]['lr']
        
        norm = 0
        for p in net.parameters():
            if p.grad is not None:
                paramNorm = p.grad.data.norm(2)
                norm += paramNorm.item()**2
        self.grad[e % step] = norm**0.5

        if e % step == 0:
            meanLoss = np.mean(self.loss)
            meanlr   = np.mean(self.lr)
            meanGrad = np.mean(self.grad)

            self.last = (f"{e}"
                         f"\tmeanL: {meanLoss:.4e}"
                         f"\tlr: {meanlr:.0e}"
                         f"\tGrad: {meanGrad:.2f}")
            print(self.last) # flush=True,

def buildSolPODNN(net,m0,m1):

    m0 = torch.tensor(np.array(m0).reshape(-1,1)).float().to(device)
    m1 = torch.tensor(np.array(m1).reshape(-1,1)).float().to(device)

    if device == 'cpu':
        u = net(m0,m1).detach().numpy().reshape(-1,)
    else:
        u = net(m0,m1).cpu().detach().numpy().reshape(-1,)

    return u

    # Reference
    # x = np.concatenate((dofs[0,:],strongs[0,:]),axis=0)
    # y = np.concatenate((dofs[1,:],strongs[1,:]),axis=0)
    # z = np.concatenate((u_k,u_strong), axis=0)