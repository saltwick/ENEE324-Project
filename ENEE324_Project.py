#!/usr/bin/env python
# coding: utf-8

# 
# This program uses data from the 2014-2018 NBA seasons to classify players by 
# position.
# 
# 
# Previous year's game statistics are used to generate a likelihood function
# for each position. Then a certain player can be tested against each of these 
# functions to guess their position.
# 

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from enum import Enum
from mpl_toolkits import mplot3d





# In[3]:



# Defines an enumeration for player positions

class Position(Enum):
  PG = 0
  SG = 1
  SF = 2
  PF = 3
  C = 4
  

# Defines an array of priors (each position's percentage of players)
# TODO: Update with actual priors
prior = {
    Position.PG : 0.2,
    Position.SG : 0.2,
    Position.SF : 0.2,
    Position.PF : 0.2,
    Position.C : 0.2 
}






# In[ ]:





# In[4]:


# Load Data
pg_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20PG-clean.csv'
sg_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20SG-clean.csv'
sf_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20SF-clean.csv'
pf_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20PF-clean.csv'
c_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20C-Clean.csv'
test_url = 'https://raw.githubusercontent.com/ssaltwick/ENEE324-Project/master/data/Data%20-%20Test-Clean.csv'

test_data = pd.read_csv(test_url).dropna()
#2p%, 3pa, ft%
# All stats
# stats = ['3PA', '3P%','2P%', 'PTS', 'AST', 'FT%', 'STL', 'BLK', 'TRB']

# Selected Stats
stats = ['3PA', '2P%',  'BLK']

# Remaining Stats
#stats = ['PTS', 'AST', 'STL', 'BLK', 'TRB']
data = {
    Position.PG : pd.read_csv(pg_url).dropna()[stats],
    Position.SG : pd.read_csv(sg_url).dropna()[stats],
    Position.SF : pd.read_csv(sf_url).dropna()[stats],
    Position.PF : pd.read_csv(pf_url).dropna()[stats],
    Position.C : pd.read_csv(c_url)[stats]
}





# In[5]:


# TODO: Generate actual MEAN and COV for each position

avgs = {
    Position.PG : data[Position.PG].mean(0).to_numpy(),
    Position.SG : data[Position.SG].mean(0).to_numpy(),
    Position.SF : data[Position.SF].mean(0).to_numpy(),
    Position.PF : data[Position.PF].mean(0).to_numpy(),
    Position.C : data[Position.C].mean(0).to_numpy()
}

covs = {
    Position.PG : data[Position.PG].cov().to_numpy(),
    Position.SG : data[Position.SG].cov().to_numpy(),
    Position.SF : data[Position.SF].cov().to_numpy(),
    Position.PF : data[Position.PF].cov().to_numpy(),
    Position.C : data[Position.C].cov().to_numpy()
}

# Display stat with minimum variance 
print(np.argmin(np.diag(covs[Position.PG])))
print(np.argmin(np.diag(covs[Position.SG])))
print(np.argmin(np.diag(covs[Position.SF])))
print(np.argmin(np.diag(covs[Position.PF])))
print(np.argmin(np.diag(covs[Position.C])))






# In[6]:


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(data[Position.PG]['3PA'],data[Position.PG]['2P%'],data[Position.PG]['BLK'])
plt.show()
plt.close(fig)
del fig
"""
# your ellispsoid and center in matrix form
A = covs[Position.PG]
center = avgs[Position.PG]

# find the rotation matrix and radii of the axes
U, s, rotation = np.linalg.svd(A)
radii = 1.0/np.sqrt(s)

# now carry on with EOL's answer
u = np.linspace(0.0, 2.0 * np.pi, 100)
v = np.linspace(0.0, np.pi, 100)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
for i in range(len(x)):
    for j in range(len(x)):
        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center



ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
plt.show()
plt.close(fig)
del fig
"""

# In[7]:


"""
  Evaluates a player against a position's likelihood.
  params: positon = Position Enum
          player = numpy array of stats

"""
def evaluate_likelihood(position, player):
  mu = avgs[position]
  sig = covs[position]
  
  p = (math.sqrt(2*math.pi)) ** 3
  c = 1 / (p * math.sqrt(np.linalg.det(sig)))
  
  # print(sig.shape, mu.shape, player.shape)
   
  t = np.dot(np.transpose(player-mu), np.dot(sig, (player-mu)))
  
  
  return c * math.exp(-0.5 * t)


# In[8]:


def guess_position(player):
  positions = {}
  positions[Position.PG] = evaluate_likelihood(Position.PG, player)
  positions[Position.SG] = evaluate_likelihood(Position.SG, player)
  positions[Position.SF] = evaluate_likelihood(Position.SF, player)
  positions[Position.PF] = evaluate_likelihood(Position.PF, player)
  positions[Position.C] = evaluate_likelihood(Position.C, player)
  # print(positions)
  v = list(positions.values())
  k = list(positions.keys())
  
  return k[v.index(max(v))]


# In[9]:


def compare_position(actual, guessed):

  pos_names = {
      Position.PG : "PG",
      Position.SG : "SG",
      Position.SF : "SF",
      Position.PF : "PF",
      Position.C : "C",
  }
      

  if pos_names[guessed] == actual:
    # print("Position %s Guessed Correctly" % actual)
    return 1
  else:
    # print("Position Guessed Incorrectly- Actual: %s     Guessed: %s" %(actual, pos_names[guessed]))
    return 0
      


# In[10]:




test_frame = test_data[['Pos'] + stats]

sample_size = test_frame.shape[0]
num_correct = 0.0
for i in range(0,sample_size):
  
  test_player = test_frame.to_numpy()[i,0:]
  
  guess = guess_position(test_player[1:])

  num_correct += compare_position(test_player[0], guess)

percent_correct = (num_correct / sample_size)
print('Guessed {:.2%}  of players correct'.format(percent_correct))


# In[11]:


print(stats)

