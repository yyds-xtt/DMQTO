import os
import numpy as np
import matplotlib.pyplot as plt
from .basic import effective_bandwidth

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# INFRA:    Class for holding Infrastructure variables
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class INFRA:
    def __init__(self):
        pass
 
    def to_npy(self):
        return np.hstack((self.VR, self.VE, self.DE, self.DR.flatten() ))
        
    def from_npy(self, npa):
        np.copyto(self.VR, npa[0:           0+self.A  ]                       )
        np.copyto(self.VE, npa[0+self.A   : 0+2*self.A]                       )
        np.copyto(self.DE, npa[0+2*self.A : 0+3*self.A]                       )
        np.copyto(self.DR, npa[0+3*self.A :           ].reshape(self.A,self.A))
   
    def copy_infra(self, infra):
        np.copyto(self.VR, infra.VR)
        np.copyto(self.VE, infra.VE)
        np.copyto(self.DE, infra.DE)
        np.copyto(self.DR, infra.DR)

    def get_info(self, i_dev):
        return self.DR[i_dev], self.DE[i_dev], self.VR[i_dev], self.VE[i_dev]
        
    def get_LEN(self):
        return (self.A*self.A) + self.A*3
    def render(self, P=print):
        P('ECA:\t', self.E,',', self.C,',', self.A )
        P("________________________________________________")

        rS='VE:\t'
        for i in range(self.A):
            rS+= str(round(self.VE[i],2))+"\t"
        P(rS)

        rS='VR:\t'
        for i in range(self.A):
            rS+= str(round(self.VR[i],2))+"\t"
        P(rS)

        rS='DE:\t'
        for i in range(self.A):
            rS+= str(round(self.DE[i],2))+"\t"
        P(rS)

        rS='DR:\t'
        for i in range(self.A):
            for j in range(self.A):
                rS+= str(round(self.DR[i,j],3))+"\t"
            rS+= "\n\t"
        P(rS)
        P("-------------------------------------------------")
        return
        

    


def infra_1():
    x=INFRA()
    
    x.name="infra_1"
    x.E = 1                     # no of Edge servers
    x.C = 1                     # no of Cloud servers
    x.A = x.E + x.C +1          # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    x.gw8_DR = 1 
    x.BW = np.array([
    #	#i0         e1			c2			#
    [	0,          5,  	    0,	   		], # i0
    [	0,          0,	        300,		], # e1
    [	0,          0,			0,	      	], # c2
    ], dtype='float')
    x.DRi = np.copy(x.BW)
    x.DR = effective_bandwidth( x.DRi ) * x.gw8_DR 
    #x.DRrL, x.DRrH = 0.5, 1.5
    

    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.538, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    #x.DErL, x.DErH = 0.9, 1.1
    
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    #x.VRrL, x.VRrH = 0.9, 1.1 # randomize ratio (0.6, 0.8)


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,         0.538, 		    0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    #x.VErL, x.VErH = 0.9, 1.1 # randomize ratio

    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_2():
    x=INFRA()
    x.name="infra_2"
    x.E = 3                             # no of Edge servers
    x.C = 2                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space


    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    x.gw8_DR = 1 
    x.BW = np.array([
    #	#i0         e1			e2          e3          c4          c5	   #
    [	0,          5,  	    5, 	   		 5,          0,  	    0,  ], # i0
    [	0,          0,	        300,		 0,          300,  	    0   ], # e1
    [	0,          0,	        0,	    	 300,        300,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          0,  	    300,], # e3
    [	0,          0,			0,	      	 0,          0,  	    500,], # c4
    [	0,          0,			0,	      	 0,          0,  	    0,  ], # c5
    ], dtype='float')
    x.DRi = np.copy(x.BW)
    x.DR = effective_bandwidth( x.DRi ) * x.gw8_DR 
    #x.DRrL, x.DRrH = 0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    #x.DErL, x.DErH =  0.9, 1.1
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    #x.VRrL, x.VRrH =  0.9, 1.1 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    #x.VErL, x.VErH = 0.9, 1.1 # randomize ratio
    
    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_3():

    x=INFRA()
    x.name="infra_3"
    x.E = 5                             # no of Edge servers
    x.C = 2                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    x.gw8_DR = 1 
    x.BW = np.array([
    #	#i0         e1			e2          e3          e4          e5          c6          c7	   #
    [	0,          5,  	    5, 	   		 5,          5,  	    5,          0,          0,  ], # i0
    [	0,          0,	        300,		 0,          0,  	    0,          300,  	    0,  ], # e1
    [	0,          0,	        0,	    	 300,        0,  	    0,          300,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          300,  	    0,          300,  	    0,  ], # e3
    [	0,          0,			0,	      	 0,          0,  	    300,        300,  	    0,  ], # e4
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    300,  ], # e5
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    500,  ], # c6
    [	0,          0,			0,	      	 0,          0,  	    0,          0,  	    0,  ], # c7
    ], dtype='float')
    x.DRi = np.copy(x.BW)
    x.DR = effective_bandwidth( x.DRi ) * x.gw8_DR 
    #x.DRrL, x.DRrH = 0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    #x.DErL, x.DErH =  0.9, 1.1
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/5.05,       1/5,       1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    #x.VRrL, x.VRrH =  0.9, 1.1 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    #x.VErL, x.VErH = 0.9, 1.1 # randomize ratio
    
    #-=============================================
    return x
    #-=============================================
    
    
def infra_4():

    x=INFRA()
    x.name="infra_4"
    x.E = 8                             # no of Edge servers
    x.C = 3                             # no of Cloud servers
    x.A = x.E + x.C +1                  # action space

    # 1 [DR] Data Rate(delay) # DR contains delay per unit of data
    x.gw8_DR = 1 
    x.BW = np.array([
    #	#i0         e1			e2          e3          e4          e5          e6              e7              e8              c9          c10          c11	   #
    [	0,          5,  	    5, 	   		 5,          5,  	    5,          5,              5,              5,              0,          0,          0,  ], # i0
    [	0,          0,	        300,		 0,          0,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e1
    [	0,          0,	        0,	    	 300,        0,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e2
    [	0,          0,	        0,	    	 0,          300,  	    0,          0,              0,              0,             300,         0,  	    0,  ], # e3
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,  	    300,         0,  ], # e4
    [	0,          0,			0,	      	 0,          0,  	    0,          300,            0,              0,              0,        300,  	    300,  ], # e5
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              300,            0,              0,          0,  	    300,  ], # e6
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              300,              0,          0,  	    300,  ], # e7
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    300,  ], # e8
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          500,  	    0,  ], # c9
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    500,  ], # c10
    [	0,          0,			0,	      	 0,          0,  	    0,          0,              0,              0,              0,          0,  	    0,  ], # c11
    ], dtype='float')
    x.DRi = np.copy(x.BW)
    x.DR = effective_bandwidth( x.DRi ) * x.gw8_DR 
    #x.DRrL, x.DRrH = 0.5, 1.5


    # 2 [DE] Data Eng # DE contains energy cost per unit of data
    gw8_DE = 1
    ar_DE = np.array ([0.56,        0.522, 		0.521, 		0.522, 		0.521, 		0.522, 		0.522, 		0.521, 		0.522, 		0.55, 		0.55, 		0.55, 		], dtype='float')
    w8_DE = np.array ([	1,         1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_DE
    x.DE = np.array(np.multiply(ar_DE, w8_DE)) 
    x.DEi = np.copy(x.DE)
    #x.DErL, x.DErH =  0.9, 1.1
    
    # 3 [VR] Task Rate(delay) # VR contains delay per unit of computation
    gw8_VR = 1
    x.VR =  np.array([	1/4.5,       1/5,       1/5.05,       1/5,       1/5.05,       1/5.05,       1/5,       1/5.05,       1/5,       1/10,		1/10,		1/10,		], dtype='float') * gw8_VR
    x.VRi = np.copy(x.VR)
    #x.VRrL, x.VRrH = 0.9, 1.1 # randomize ratio


    # 4 [VE] Task Eng # # VE contains energy cost per unit of computation
    gw8_VE = 1
    ar_VE = np.array ([0.56,       0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.52, 		 0.55, 		0.55, 		0.55, 		], dtype='float')
    w8_VE = np.array ([1,          1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    1,		    ], dtype='float') * gw8_VE
    x.VE = np.array(np.multiply(ar_VE, w8_VE)) 
    x.VEi = np.copy(x.VE)
    #x.VErL, x.VErH =  0.9, 1.1 # randomize ratio
    
    #-=============================================
    return x
    #-=============================================
    
    
PRE_BUILT_INFRA = {
    
    'infra_1': infra_1,
    'infra_2': infra_2,
    'infra_3': infra_3,
    'infra_4': infra_4,

        }










# Notes
