#!/usr/bin/python

"""
   This script generates
   files the files with data structures 
   of boolean network used by Biocellion. 
   The data structures are very similar to those used
   in PBN (Probabilitic Boolan Network) matab toolbox developed
   by Ilya Shmulevich
   
   The script will generate
   .nv file with the number of variables per gene.
   .varf file with the ides of the inputs for each node
   .tt file  with the truth table 

   Usage:
   python  generate_truth_tables.py  -i <inputfile> -o <outputfile>

   <inputfile> is a file with all the boolean rules (see examples)
   Files will have the names <outputfile>.nv <outputfile>.varf <outputfile>.tt  
   
""" 

import sys, getopt
import numpy 
from sympy import *
import re

####### PrintVector()
# Print a vector of numbers in the screen
#
# Parameters :
#    Vector  : array of integers
#    OutFile : name of the output file
#
# Called by: Main()
# Calls:  none 
def PrintVector( Vector , OutFile  ):
    N = len(Vector)
    for i in range(N):
        OutFile.write("%s " % str(Vector[i])) 
#        s = s + str(Vector[i])+ " ",
#    OutFile.write( s + "\n" ) 


####### main()
# Main function
def main( argv ) :
   infileName = ''
   outfileName = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print( 'generate_truth_tables.py  -i <inputfile> -o <outputfile>')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print( 'generate_truth_tables.py  -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         infileName = arg
      elif opt in ("-o", "--ofile"):
         outfileName = arg

   if ( infileName == '' ) or ( outfileName == '' ) :
      print ('generate_truth_tables.py -i <inputfile> -o <outputfile>')
      sys.exit()


   TimeScale  = []
   Names = [] 
   RightHS = []
   BN_NUM_NODES = 0;


   # Read line by line, saving the time sclae, thee names of the nodes 
   # and the rhs of the equations
   infile = open( infileName , "r" )
   for line in infile:
      # Every boolean function need a timescle ( tau = scale)
      temp_names = line.strip().split('tau')
      if ( len(temp_names) !=  2):
         print("ERROR: Missing time scale for this Function at line or too many values after tau ", BN_NUM_NODES+1)
         print( len(temp_names))
         sys.exit()  

      TimeScale.append( int(temp_names[1]) )
    
      left_rigth_side  = temp_names[0].strip().split('=') 
      if (len(left_rigth_side) != 2 ):
         print( "ERROR: sript is unable to read the Boolean function at line",  BN_NUM_NODES+1 )
         sys.exit()
  
      # Save names and the actual boolean functions 
      Names.append( left_rigth_side[0].strip() ) 
      RightHS.append( left_rigth_side[1] )

      BN_NUM_NODES  = BN_NUM_NODES + 1

   infile.close()

   # Checking possible problems
   if (len(TimeScale) != len(Names)):
      print( "error : problems with the input file Num Names != N time scales" )
      sys.exit()

   # create a map of names and indexes
   Name_map = {}
   for i in range(BN_NUM_NODES):
      Name_map[Names[i]] = i
 
   # Number of function per node,
   BN_NUM_F = numpy.zeros( BN_NUM_NODES,  dtype=numpy.int8 )
   BN_TOTAL_NF =0
   for i in range( BN_NUM_NODES) :
      if (TimeScale[i] == 0 ) : BN_NUM_F[i] = 1 
      else : BN_NUM_F[i] = TimeScale[i] 
      BN_TOTAL_NF = BN_TOTAL_NF + BN_NUM_F[i]

   # this array will be helpul latter
   SUM_CUM = numpy.zeros(BN_NUM_NODES, dtype=numpy.int8 ) 
   SUM_CUM[0] = 0;
   for i in range(BN_NUM_NODES):
      if (i>0): SUM_CUM[i] = SUM_CUM[i-1] + BN_NUM_F[i-1] 

   # compute the Max indgree
   BN_MAX_INDEGREE  = 0
   BN_NVAR = numpy.zeros( BN_TOTAL_NF , dtype=numpy.int8 )

   for gene in range(BN_NUM_NODES) :
 
      rhs = RightHS[gene]
      # Replce operators and parenthesis by spaces
      # strings other than |,&,~,(,), will be interpreted as names.
      temp_names = rhs.replace("|"," ").replace("&"," ").replace("~"," ").replace("("," ").replace(")"," ")
      temp_hash ={}
      for inputs in temp_names.split():
         if ( inputs in Name_map  ): #check the string is a node name
            temp_hash[inputs] = 1
         elif ( (inputs == "0") | (inputs == "1") ): continue
         else :
            print( "ERROR: the following string is not part of the node names:" , inputs )
            sys.exit()

      if ( len( temp_hash )  > BN_MAX_INDEGREE): # update maximum as standard
         BN_MAX_INDEGREE = len( temp_hash )
   
      Indx = SUM_CUM[gene] + BN_NUM_F[gene] - 1
      BN_NVAR[Indx] = len( temp_hash )  

   BN_MAX_NSTATES  = 2**BN_MAX_INDEGREE # This is for boolean only


   # Generate the inputs for the variables
   BN_VAR_F = -numpy.ones(( BN_MAX_INDEGREE,BN_TOTAL_NF   ), dtype=numpy.int8 )
   for gene in range(BN_NUM_NODES):

      rhs = RightHS[gene]
      temp_names = rhs.replace("|"," ").replace("&"," ").replace("~"," ").replace("("," ").replace(")"," ")
      temp_hash ={}
      for inputs in temp_names.split():
         if ( inputs in Name_map  ): #check the string is a node name
            temp_hash[inputs] = 1
         elif ( (inputs == "0") | (inputs == "1") ): continue

      Indx = SUM_CUM[gene] + BN_NUM_F[gene] - 1
    
      j = 0
      for var in temp_hash :
         BN_VAR_F[j][Indx] = Name_map[var]
         j = j + 1


   # THE TRUTH TABLE !!
   BN_F = -numpy.ones(( BN_MAX_NSTATES, BN_TOTAL_NF), dtype=numpy.int8 )
   for gene in range(BN_NUM_NODES):

      # create the boolean expression
      variables_str = ''
      expresion_str = RightHS[gene]
 
      Indx = SUM_CUM[gene] + BN_NUM_F[gene] - 1

      for j in range(BN_NVAR[Indx] ):
          k = BN_VAR_F[j][Indx]
          node_id  = 'x' + str( k )
          name = Names[k]
          #tmp_str = expresion_str.replace( name , node_id )
          tmp_str = re.sub(rf"{name}(?=\s|\)|$)", node_id, expresion_str  )
          expresion_str = tmp_str

     
      for j in range(BN_NVAR[Indx] ):
          k =  BN_VAR_F[j][Indx]
          variables_str = variables_str + 'x' + str(BN_VAR_F[j][Indx]) +  ' '

      if ( BN_NVAR[Indx] == 1 ):
         expr = sympify( expresion_str )
         variables =  symbols( variables_str )
         exp_hash = {};
         exp_hash[variables] = False
         BN_F[0][Indx] =  1 if expr.subs( exp_hash ) else 0
         exp_hash[variables] = True
         BN_F[1][Indx]  = 1 if expr.subs( exp_hash ) else 0
    
      elif ( BN_NVAR[Indx] > 1 ):
         expr = sympify( expresion_str  )
         variables =  symbols( variables_str )
         counter = 0
         for truth_values in cartes([False, True], repeat = BN_NVAR[Indx] ) :
            values = dict(zip(variables, truth_values))
            BN_F[counter][Indx] =  1 if  expr.subs(values) else 0
            counter = counter + 1

   
   # print the nv file (number of inputs per node)
   nvFile = open( outfileName + '.nv' , 'w' )
   PrintVector(BN_NVAR , nvFile )
   nvFile.close()

   # print varF files, ids of input nodes for each node
   # add 1 to the indexes for compatibility with matlab
   for i in range( BN_MAX_INDEGREE ): 
      for j in range ( BN_NUM_NODES ) :
           if ( BN_VAR_F[i,j] > -1  ) :
              BN_VAR_F[i,j] = BN_VAR_F[i,j] + 1 
   # print 
   varfFile = open( outfileName + '.varf' , 'w' )
   for i in range( BN_MAX_INDEGREE ):
      PrintVector( BN_VAR_F[i,:]  , varfFile )
      varfFile.write("\n")
   varfFile.close()

   # print truth table (tt)
   ttableFile = open( outfileName + '.tt' , 'w' ) 
   for i in range( BN_MAX_NSTATES ):
      PrintVector( BN_F[i,:], ttableFile  )
      ttableFile.write("\n")
   ttableFile.close()

   #sys.exit()
if __name__ == "__main__":
   main(sys.argv[1:])

