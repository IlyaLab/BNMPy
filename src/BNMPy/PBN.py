import os
import numpy as np
import pandas as pd


class ProbabilisticBN(object):

    def __init__(self, numberOfNodes, linkages, numberOfFunctions, functions, probabilities, initialNodeValues,
                 outputFilePath=''):
        self.N = numberOfNodes
        self.varF = np.array(linkages)
        self.nf = np.array( numberOfFunctions )
        self.nf_max = np.max( self.nf )
        self.F = np.array(functions, dtype=np.int8)
        self.nodes = np.array(initialNodeValues, dtype=np.int8)
        self.networkHistory = np.array([[node for node in self.nodes]])
        # print('after nodes are added to networkHistory')
        
        self.cij = np.array( probabilities )

        self.K = []
        self.isConstantConnectivity = True
        self.isConstanNode = np.full( self.N , False)
        
        # helper array to obtained correc indexes for functions
        temp = np.cumsum( self.nf ) 
        self.cumsum = np.insert(temp,0,0)
        
        self.Nf = np.sum( self.nf ) 
        

        # Obtain the number of inputs per function
        for i in range( self.Nf ):
            
            self.K.append(0)
            for linkage in self.varF[ i ]:
                if (linkage == -1):
                    self.isConstantConnectivity = False
                    break
                self.K[i] += 1
        self.K = np.array(self.K)
        
        # determine if a node is constant
        for i in range( self.N ) :
            idx = self.cumsum[i]            
            if ( self.K[idx] == 0 ) and ( self.Nf[i] == 1 ) :
                self.isConstanNode[i] = True
        
        
        self.outputFilePath = outputFilePath

        if ( outputFilePath != '' ) :
            self.initializeOutput()

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
                
                if ( self.isConstanNode[i] ) :
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

    # update the Boolean network with noise, derived from update functions (Boris)
    def update_noise(self, p, iterations=1 ): 
        
        y = np.zeros( (iterations + 1, self.N ) , dtype=np.int8  )
        temp = np.array(  [2**i for i in range(self.N-1, -1, -1) ] ) 
        
        y[0] = self.nodes
        
        for itr in range(iterations):
            
            # check if there is a better way to do this [[ Boris ]] 
            gam = np.array( [ False if self.isConstanNode[i] else np.random.rand() < p  for i in range(self.N)  ])
            
            if np.any(gam):
                y[itr+1] =  np.bitwise_xor(  y[itr]  ,  gam)  
                
            else :
                for i in range(self.N):
                
                    if ( self.isConstanNode[i] ) :
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

