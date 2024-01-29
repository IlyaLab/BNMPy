import os
import numpy as np
import pandas as pd


class BooleanNetwork(object):

    def __init__(self, numberOfNodes, linkages, functions, initialNodeValues,
                 outputFilePath=''):
        self.N = numberOfNodes
        self.varF = np.array(linkages)
        self.F = np.array(functions, dtype=np.int8)
        self.nodes = np.array(initialNodeValues, dtype=np.int8)
        self.networkHistory = np.array([[node for node in self.nodes]])
        # print('after nodes are added to networkHistory')

        self.K = []
        self.isConstantConnectivity = True
        self.isConstanNode = np.full( self.N , False)
        for i in range(len(self.varF)):
            self.K.append(0)
            for linkage in self.varF[i]:
                if (linkage == -1):
                    self.isConstantConnectivity = False
                    break
                self.K[i] += 1
        
            if self.K[i] == 0 :
                self.isConstanNode[i] = True
                
        self.K = np.array(self.K)

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
        states = []
        for iterationNumber in range(iterations):
            newNodes = []
            for i in range(self.N):
                # print('N = %d, K[%d] = %d' % (i, i, self.K[i]))
                if (self.K[i] > 0):
                    fInput = 0
                    for j in range(self.K[i]):
                        # print('j = %d, self.varF[%d][%d] = %d' % (j, i, j, self.varF[i,j]))
                        # print('self.nodes[self.varF[%d][%d]] = %d' % (i, j, self.nodes[self.varF[i][j]]))
                        assert self.varF[i,j] >= 0

                        fInput += self.nodes[self.varF[i,j]] * (2 **  # check varF
                                                                 (self.K[i]
                                                                  - 1 - j))
                    newNodes.append(self.F[i,fInput])
                else:
                    newNodes.append(self.nodes[i])
                # print('newNodes = ', newNodes)

            self.nodes = np.array(newNodes, dtype=np.int8)

            history = self.networkHistory.tolist()
            history.append([self.nodes[index] for index in range(self.N)])
            self.networkHistory = np.array(history, dtype=np.int8)
            # print('network nodes: ', [self.nodes[i] for i in range(N)], '\n')
            # print(self.networkHistory, '\n\n\n')

            # note that these four lines take about 90% of the computing time
            # file = open(self.outputFilePath,'a')
            # str = self.stateToWrite()
            # file.write(str)
            # file.close()

    # update the Boolean network with noise, derived from update functions (Boris)
    def update_noise(self, p, iterations=1 ): 
        states = []
        for iterationNumber in range(iterations):
            newNodes = []
            
            # check if there is a better way to do this [[ Boris ]] 
            gam = np.array( [ False if self.isConstanNode[i] else np.random.rand() < p  for i in range(self.N)  ])
            
            if np.any(gam):
                
                newNodes =  np.bitwise_xor(  self.nodes ,  gam)  
                
            else :
                for i in range(self.N):
                    # print('N = %d, K[%d] = %d' % (i, i, self.K[i]))
                    if (self.K[i] > 0):
                       fInput = 0
                       for j in range(self.K[i]):
                           # print('j = %d, self.varF[%d][%d] = %d' % (j, i, j, self.varF[i,j]))
                           # print('self.nodes[self.varF[%d][%d]] = %d' % (i, j, self.nodes[self.varF[i][j]]))
                           assert self.varF[i,j] >= 0
        
                           fInput += self.nodes[self.varF[i,j]] * (2 **  # check varF
                                                                    (self.K[i]
                                                                     - 1 - j))
                       newNodes.append(self.F[i,fInput])
                    else:
                       newNodes.append(self.nodes[i])
                       # print('newNodes = ', newNodes)

            self.nodes = np.array(newNodes, dtype=np.int8)

            history = self.networkHistory.tolist()
            history.append([self.nodes[index] for index in range(self.N)])
            self.networkHistory = np.array(history, dtype=np.int8)

 	

    # Updates the network until either an attractor is reached (in which case
    # it returns the number of updates it took from the original state of the
    # network to reach an attractor), or, if it doesn't confirm it reached an
    # attractor in giveUpIterations steps, it returns numpy.NaN
    def updateUntilAttractorIsReached(self, giveUpIterations):
        for _ in range(giveUpIterations):
            #df = pd.read_csv(self.outputFilePath)       # change to look at network history
            for index in range(len(self.networkHistory)):
                for testState in self.networkHistory[index+1:]:
                    # print('%d, %d' % (index, testIndex))
                    # print(df.iloc[index] - df.iloc[testIndex])
                    if not (self.networkHistory[index] - testState).any():
                        return index
            self.update()
        return np.NaN

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



def getRandomParameters(numberOfNodes, connectivity,
                        isConstantConnectivity=True, bias=0.5):
    if numberOfNodes < connectivity:
        raise ValueError('Connectivity larger than number of nodes')

    # Generate connectivities
    connectivities = []
    if (isConstantConnectivity):
        for _ in range(numberOfNodes):
            connectivities.append(connectivity)
        maxConnectivity = connectivity
    else:
        # Note that this is a specific distribution (uniform)
        # of the values of K[i]
        for _ in range(numberOfNodes):
            connectivities.append(0)
        for _ in range(int(connectivity * numberOfNodes)):
            connectivities[np.random.randint(0, numberOfNodes)] += 1

        maxConnectivity = max(connectivities)

    # Generate linkages
    linkages = []
    for i in range(numberOfNodes):
        linkages.append([])
        for _ in range(connectivities[i]):
            while True:
                newnode = np.random.randint(0, numberOfNodes)
                if newnode not in linkages[i]:
                    linkages[i].append(newnode)
                    break

        for _ in range(maxConnectivity - connectivities[i]):
            linkages[i].append(-1)

    # Generate functions
    functions = []
    for i in range(numberOfNodes):
        functions.append([])
        # Initialize a maxK by N matrix filled with -1
        for _ in range(2 ** maxConnectivity):
            functions[i].append(-1)
        # Fill in K[i] values
        for j in range(2 ** connectivities[i]):
            if np.random.random() < bias:
                functions[i][j] = 1
            else:
                functions[i][j] = 0

    # Initialize nodes
    initialNodeValues = []
    for _ in range(numberOfNodes):
        initialNodeValues.append(np.random.randint(2))

    return (linkages, functions, initialNodeValues)

def getRandomInitialNodeValues(numberOfNodes):
    initialNodeValues = []
    for _ in range(numberOfNodes):
        initialNodeValues.append(np.random.randint(2))

    return initialNodeValues

def getDataFromFile(filepath, startIndex=0, invert=False):
    file = open(filepath)
    rawData = file.readlines()
    file.close()

    data = []
    for i in range(1, len(rawData)):
        data.append([])
        for split in rawData[i].split(','):
            try:
                datum = int(split)
                data[i - 1].append(datum)
            except ValueError:
                pass

    data = data[startIndex:]

    if not invert:
        if len(data) == 1:
            return data[0]
        # elif len(data[0]) == 1:
        #     for i in range(len(data)):
        #         data[i] = data[i][0]
        return data

    invertedData = []
    for i in range(len(data[0])):
        invertedData.append([])
        for j in range(len(data)):
            invertedData[i].append(data[j][i])

    return invertedData


def getParametersFromFile(numberOfNodes, linkagesFilePath,
                          functionsFilePath, initialNodesFilePath):
    # Get data from files
    linkages = getDataFromFile(linkagesFilePath, invert=True)
    functions = getDataFromFile(functionsFilePath, invert=True)
    initialNodeValues = getDataFromFile(initialNodesFilePath, invert=False)

    # Change inputs from being one-indexed to zero-indexed
    for link in range(len(linkages)):
        for input in range(len(linkages[link])):
            if linkages[link][input] > 0:
                linkages[link][input] -= 1

    if len(linkages) != numberOfNodes or len(functions) != numberOfNodes \
            or len(initialNodeValues) != numberOfNodes:
        raise ValueError(
            'Number of nodes does not match network parameter(s)')

    return (linkages, functions, initialNodeValues)

    # Test above before implementing for functions or initial nodes

    # self.__init__(numberOfNodes, linkages, functions, initialNodes)


#directory = os.getcwd() + '/'
#linkages_filename = 'linkages_test1.csv'
#functions_filename = 'functions_test1.csv'
#initial_filename = 'init_test1.csv'


#(varF, F, init) = getParametersFromFile(3, directory + linkages_filename,
#                                           directory + functions_filename,
#                                           directory + initial_filename)


#ngenes = 3

#varF = np.array( [[0, 1, -1],  # indices of genes connected to gene 0
#                  [0, 1, -1],  # indices of genes connected to gene 1
#                  [0, 1, 2]] ) # indices of genes connected to gene 2

#F = np.array( [[0, 1, 1, 1, -1, -1, -1, -1], # truth table for gene 0 
#               [0, 1, 1, 1, -1, -1, -1, -1], # truth table for gene 1
#               [0, 0, 1, 0, 1, 0, 1, 0]] ) # truth table for gene 2


#x0  =  np.array( [0, 1, 1] )
#network = BooleanNetwork( ngenes , varF, F, x0 , outputFilePath = 'test_2018-06-26_2.csv')
#network = BooleanNetwork( ngenes , varF, F, x0  )
#network.update_noise (0.01  , 20)
#network.update( 3 )

#network.writeNetworkHistory()


#y = network.getTrajectory() 

#p = np.array(  [2**i for i in range(ngenes-1, -1, -1) ] )  

#states = [ yi @ p.T for yi in y ] 

#print( states )

#print ( " --- " )
#print(y)

