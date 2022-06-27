#Othello game :

This repository is an Otello game with **AI opponent**. There is a graphical interface that allows you to visualize the game.
The aim of the game is to turn over the opponent's chips on a checkerboard.

## Code info :
The chosen algorithm is **'alpha-beta'**, the search depth can change depending on the state of the game.  

> Library used: 
  - TKinter 
  - NUMPY 
  - NUMBA
--- 
The Scoring is the following : 
- *Start of game* : maximise the number of places possible to be more successful in the next phase which is key
- *Middle of game* :  maximise your pieces on the edges and especially the corners as these are more strategic positions
- *End of game* :maximising points, here we can do a classical allocation of points because we can simulate until the end
--- 

### Global info : 
This is a **university project** made in pairs with @Benoit-Marchadier
