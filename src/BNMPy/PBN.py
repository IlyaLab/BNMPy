import os
import numpy as np
import pandas as pd


class ProbabilisticBN(object):

    def __init__(self, numberOfNodes, linkages, numberOfFunctions, functions, probabilities, initialNodeValues,
                 outputFilePath='', nodeDict=None):
        
        self.N = numberOfNodes
        self.varF = np.array(linkages)
        self.nf = np.array( numberOfFunctions )
        self.cij = np.array( probabilities )
        self.F = np.array(functions, dtype=np.int8)
        self.nodes = np.array(initialNodeValues, dtype=np.int8)

        self.networkHistory = np.array([[node for node in self.nodes]])
        self.nf_max = np.max( self.nf )
        self.K = []
        
        # helper array to obtained correc indexes for functions
        temp = np.cumsum( self.nf ) 
        self.cumsum = np.insert(temp,0,0)
        
        self.Nf = np.sum( self.nf ) 
        self.isConstanNode = np.full( self.Nf , False)    

        # Obtain the number of inputs per function
        for i in range( self.Nf ):
            self.K.append(0)
            for linkage in self.varF[ i ]:
                if (linkage == -1):
                    break
                self.K[i] += 1
        self.K = np.array(self.K)
        
        # determine if a node is constant
        for i in range( self.Nf ) :
            if ( self.K[i] == 0 )  :
                self.isConstanNode[i] = True
        
        self.outputFilePath = outputFilePath

        if ( outputFilePath != '' ) :
            self.initializeOutput()
        
        # Add nodeDict attribute for gene name to index mapping
        if nodeDict is None:
            nodeDict = {i: i for i in range(self.N)}
        self.nodeDict = nodeDict        

        # old connectivity matrices - for mutations
        self.old_varF = None
        self.old_F = None
        self.old_nf = None
        self.old_cij = None
        self.old_cumsum = None

    def buildK(self):
        "This rebuilds the K array and related attributes."
        self.K = []
        
        # Rebuild K array
        for i in range(self.Nf):
            self.K.append(0)
            for linkage in self.varF[i]:
                if (linkage == -1):
                    break
                self.K[i] += 1
        self.K = np.array(self.K)
        
        # Rebuild isConstanNode array
        self.isConstanNode = np.full(self.Nf, False)
        for i in range(self.Nf):
            if (self.K[i] == 0):
                self.isConstanNode[i] = True
    
    def setInitialValues(self, initialNodeValues):
        "Sets the initial values of the probabilistic boolean network."
        self.nodes = np.array(initialNodeValues, dtype=np.int8)

    def setInitialValue(self, key, value):
        "Sets a particular node to a given initial value, where the key is indexed in nodeDict."
        ind = self.nodeDict[key]
        self.nodes[ind] = value

    def knockout(self, key, value):
        "Sets a specific node to be permanently fixed to a given value."
        if self.old_varF is None:
            self.old_varF = self.varF.copy()
            self.old_F = self.F.copy()
            self.old_nf = self.nf.copy()
            self.old_cij = self.cij.copy()
            self.old_cumsum = self.cumsum.copy()
            
        # Set the initial value of the node
        self.setInitialValue(key, value)
        
        # Get the node index
        node_idx = self.nodeDict[key]
        
        # Set the number of functions for this node to 1
        self.nf[node_idx] = 1
        
        # Update cumsum array
        temp = np.cumsum(self.nf)
        self.cumsum = np.insert(temp, 0, 0)
        
        # Update Nf
        self.Nf = np.sum(self.nf)
        
        # Set the probability for this node's single function to 1.0
        for i in range(len(self.cij[node_idx])):
            if i == 0:
                self.cij[node_idx, i] = 1.0
            else:
                self.cij[node_idx, i] = -1.0
                
        # Find the function index for this node
        func_idx = self.cumsum[node_idx]
        
        # Set the connectivity of this node to -1 (indicating constant)
        self.varF[func_idx, :] = -1
        
        # Set the truth table entry to the constant value
        self.F[func_idx, 0] = value
        for i in range(1, len(self.F[func_idx])):
            self.F[func_idx, i] = -1
            
        # Rebuild K array and related attributes
        self.buildK()

    def undoKnockouts(self):
        "Undoes all knockouts. Does not change initial values, however."
        if self.old_varF is not None:
            self.varF = self.old_varF
            self.F = self.old_F
            self.nf = self.old_nf
            self.cij = self.old_cij
            self.cumsum = self.old_cumsum
            self.Nf = np.sum(self.nf)
            
            # Reset stored originals
            self.old_varF = None
            self.old_F = None
            self.old_nf = None
            self.old_cij = None
            self.old_cumsum = None
            
            # Rebuild K array and related attributes
            self.buildK()

    def initializeOutput(self):
        file = open(self.outputFilePath, 'w')
        stringToWrite = ''
        for i in range(self.N):
            stringToWrite += 'Node_{},'.format(i + 1)
        # replace last comma with newline
        stringToWrite = stringToWrite[:-1] + '\n'
        file.write(stringToWrite)
        file.write(self.stateToWrite())
        file.close()

    def stateToWrite(self):
        stringToWrite = ''
        for node in self.nodes:
            stringToWrite += (str(node) + ',')
        stringToWrite = stringToWrite[:-1] + '\n'

        return stringToWrite

    def writeNetworkHistory(self):
        string = ''
        for timestep in self.networkHistory:
            for node in timestep:
                string += (str(node) + ',')
            string = string[:-1] + '\n'
        file = open(self.outputFilePath, 'w')
        file.write(string)
        file.close()

    def update(self, iterations=1):
        
        y = np.zeros( (iterations + 1, self.N ) , dtype=np.int8  )
        temp = np.array(  [2**i for i in range(self.N-1, -1, -1) ] ) 
        
        y[0] = self.nodes
        
        for itr in range(iterations):

            for i in range(self.N):
                
                cnf = self.cij[i,0:self.nf[i]]               
                idx = self.cumsum[i] + np.random.choice( self.nf[i], 1, p=cnf)[0] 
                    
                fInput = 0
                for j in range(self.K[idx]):    
                    fInput += (y[itr][ self.varF[idx,j]]) * temp[ j - self.K[idx]  ]
                    
                y[itr+1][i] = self.F[idx,fInput]

        self.nodes = y[-1] # newNodes

        return y
    
    # update the Boolean network with noise, derived from update functions (Boris)
    def update_noise(self, p, iterations=1 ): 
        
        y = np.zeros( (iterations + 1, self.N ) , dtype=np.int8  )
        temp = np.array(  [2**i for i in range(self.N-1, -1, -1) ] ) 
        
        y[0] = self.nodes
    
        # Create a mask for nodes that can be affected by noise
        # Nodes that have been knocked out should not be affected by noise
        knocked_out_nodes = np.zeros(self.N, dtype=bool)
        
        # Identify knocked out nodes by checking if they have exactly 1 function
        # and if that function has no inputs (K=0)
        for i in range(self.N):
            if self.nf[i] == 1:
                func_idx = self.cumsum[i]
                if np.all(self.varF[func_idx] == -1):
                    knocked_out_nodes[i] = True
        
        for itr in range(iterations):
             # Only apply noise to nodes that haven't been knocked out
            noise_mask = np.logical_not(knocked_out_nodes)
            gam = np.zeros(self.N, dtype=bool)
            for i in range(self.N):
                if noise_mask[i] and np.random.rand() < p:
                    gam[i] = True
            
            if np.any(gam):
                y[itr+1] = np.bitwise_xor(y[itr], gam)  
                
            else :
                for i in range(self.N):
                    if knocked_out_nodes[i]:
                        # For knocked out nodes, just copy their value
                        y[itr+1][i] = y[itr][i]
                    else:                
                        cnf = self.cij[i,0:self.nf[i]]               
                        idx = self.cumsum[i] + np.random.choice( self.nf[i], 1, p=cnf)[0] 
                        
                        fInput = 0
                        for j in range(self.K[idx]):    
                            fInput += (y[itr][ self.varF[idx,j]]) * temp[ j - self.K[idx]  ]
                        
                        y[itr+1][i] = self.F[idx,fInput]
                    


        self.nodes = y[-1] # newNodes

        return y 


    def getRealization(self):
        return (self.F, self.varF)

    def getMeanConnectivity(self):
        return sum(self.K) / len(self.K)

    def getMaxConnectivity(self):
        return max(self.K)

    def getBias(self):
        num0 = 0
        num1 = 0
        for function in self.F:
            for value in function:
                if value == 1:
                    num1 += 1
                elif value == 0:
                    num0 += 1
        return num1 / (num0 + num1)
   
    def getTrajectory( self ) : 
        return  self.networkHistory




def getRandomInitialNodeValues(numberOfNodes):
    initialNodeValues = []
    for _ in range(numberOfNodes):
        initialNodeValues.append(np.random.randint(2))

    return initialNodeValues

