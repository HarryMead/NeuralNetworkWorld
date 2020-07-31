#!/usr/bin/env python
from cosc343worldcc import _cCreature, _cWorld
import numpy as np
import time
import sys

# This is a creature class that your EvolvingCreature needs to inherit from.
# This class wraps the _cCreature class which was implemented in C.
class Creature(_cCreature):

    # Your child class must override this method, where the
    # mapping of percepts to actions is implemented
    def AgentFunction(self, percepts, nActions):
        print("Your EvolvingCreature needs to override the AgentFunction method!")
        sys.exit(-1)

    # Agent function, which is called from the simulation of the world implemented in C.
    # This method translates the percepts to a python list, and translates back
    # the list representing the actions into C format.
    def internal_AgentFunction(self):

        # Get the number of percepts and actions
        nPercepts = self.numPercepts()
        nActions = self.numActions()

        # Create lists of percepts
        percepts = np.zeros((nPercepts))
        for i in range(nPercepts):
            percepts[i] = self.getPercept(i)

        # Execute the AgentFunction method that needs to be implemented
        # by the EvolvingCreature.  Pass in the list of percepts and
        # specify the number of actions expected.
        actions = self.AgentFunction(percepts, nActions)

        if not isinstance(actions, list) or len(actions) != nActions:
            print("Error!  Expecting the actions returned from the AgentFunction to be a list of %d numbers." % nActions)

        # Translate actions and feed it back to the engine
        for i in range(nActions):
            self.setAction(i, actions[i])

# Wrapper class for _cWorld which implements the engine for the simulation
class World(_cWorld):

   # Initialise the wrapper with some defaults for the world type, grid size
   # and the repeatability setting.
   def __init__(self, worldType=1, gridSize=24, repeatable=False):
      self.ph = None
      self.worldType = worldType
      super().__init__(worldType, gridSize, repeatable)

   # Feed the next generation of creatures to the simulation
   #
   # Input: population - a list of creatures for the simulation
   def setNextGeneration(self, population):
      self.resetCreatures()
      for i in range(len(population)):
         self.addCreature(population[i])

   # Animation of the simulation
   #
   # Input: titleStr - title string of the simulation
   #        speed - of the visualisation: can be 'slow', 'normal' or 'fast'
   def show_simulation(self, titleStr = "", speed='normal'):
      import pygame
      gridSize = self.gridSize()
      left_frame = 100

      # Initialise pygame
      pygame.init()

      # Specify the size of the widnow
      size = width, height = 720, 480
      WHITE = (255, 255, 255)
      BLACK = 0, 0, 0

      if speed == "normal":
          frameTurns = 20
          nSteps = 10
      elif speed == "fast":
          frameTurns = 1
          nSteps = 5
      elif speed == "slow":
          frameTurns = 40
          nSteps = 10

      # Create pygame screen
      screen = pygame.display.set_mode(size)

      # Compute the size of the individual square
      unit = int(np.min([width-left_frame, height])/gridSize)

      # Load images
      im_strawbs = [pygame.image.load('images/strawberry-green.png'),
                    pygame.image.load('images/strawberry-red.png')
                   ]

      im_creatures = [pygame.image.load('images/smiley_happy.png'),
                      pygame.image.load('images/smiley_hungry.png'),
                      pygame.image.load('images/smiley_sick.png')
                     ]

      # Scale the images for the size of the individual square
      for i in range(len(im_strawbs)):
          im_strawbs[i] = pygame.transform.scale(im_strawbs[i], (unit, unit))

      for i in range(len(im_creatures)):
          im_creatures[i] = pygame.transform.scale(im_creatures[i], (unit, unit))

      im_monster = pygame.transform.scale(pygame.image.load("images/monster.png"), (unit, unit))

      # Read the total number of turns from the engine
      nTurns = self.vis_numTurns()
      # The speed of animation depends on specified speed
      stepDiff = 1.0/float(nSteps)

      # Read the number food items, creatures and monsters from the engine
      nFood = self.vis_num(0)
      nCreatures = self.vis_num(1)
      nMonsters = self.vis_num(2)

      nBodies = [nFood, nCreatures, nMonsters]

      halfSteps = int(np.floor(nSteps/2))

      # Showing visulisation of the simulation state at each turn
      for t in range(1, nTurns + 1):

          # Update the window caption to specify the turn number
          pygame.display.set_caption("World %d, %s (turn %d)" % (self.worldType, titleStr, t))

          # The nSteps is the number of animations between a turn (the slower, the smoother the animation)
          for k in range(nSteps):

              for event in pygame.event.get():
                  if event.type == pygame.QUIT: sys.exit()

              # Paint the window in white
              screen.fill(WHITE)

              # Draw the grid lines in black
              for i in range(gridSize + 1):
                 pygame.draw.line(screen, BLACK, [left_frame, i*unit], [left_frame+(gridSize*unit), i*unit])
                 pygame.draw.line(screen, BLACK, [left_frame+(i*unit), 0], [left_frame+(i*unit), gridSize * unit])

              # Iterate over all item types...
              for type in range(3):
                  # For the number of items in each type...
                  for i in range(nBodies[type]):
                      # Get the position and state at turn t
                      x = self.vis(type, 0, i, t)
                      y = self.vis(type, 1, i, t)
                      s = self.vis(type, 2, i, t)

                      # Get the position at turn t-1
                      xprev = self.vis(type, 0, i, t-1)
                      yprev = self.vis(type, 1, i, t-1)

                      # Compute the shift from t-1 to t based on current frame
                      xshift = xprev-x
                      if np.abs(xshift)<=1:
                          xdiff = (x - xprev) * k * stepDiff
                      elif k <= halfSteps:
                          xdiff = np.sign(xshift) * k * stepDiff
                      else:
                          xdiff = -np.sign(xshift) * k * stepDiff
                          xprev = x

                      yshift = yprev - y
                      if np.abs(yshift) <= 1:
                          ydiff = (y - yprev) * k * stepDiff
                      elif k <= halfSteps:
                          ydiff = np.sign(yshift) * k * stepDiff
                      else:
                          ydiff = -np.sign(yshift) * k * stepDiff
                          yprev = y

                      # If the item is food...
                      if type==0:
                          # ...depending on the state show the green or red strawberry icon
                          if s >= 0 and s <= 1:
                              obj_loc = pygame.Rect(left_frame + (x * unit), y * unit, unit, unit)
                              obj_im = im_strawbs[s]
                              screen.blit(obj_im, obj_loc)

                      # If the item is a creature...
                      elif type==1:
                          # ...show only if not dead
                          if s > 0:
                              # Depending on state show different creature icon
                              obj_im = im_creatures[s-1]
                              obj_loc = pygame.Rect(left_frame + (xprev + xdiff) * unit, (yprev + ydiff) * unit, unit,
                                                    unit)
                              screen.blit(obj_im, obj_loc)

                      # If the item is a monster...
                      elif type==2:
                          #...show the monster icon
                          obj_loc = pygame.Rect(left_frame+(xprev + xdiff) * unit, (yprev + ydiff) * unit, unit, unit)
                          screen.blit(im_monster, obj_loc)

              # Update the dislplay
              pygame.display.flip()
              pygame.time.delay(frameTurns)
      pygame.display.quit()
      pygame.quit()
