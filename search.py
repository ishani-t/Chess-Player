import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove
from statistics import mean

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # base case: minimax(node) = utility(node) if the node is terminal
    # at depth 0, return evaluate(board), [], and {} 
    # ==> we return empty at bottom for moveList and moveTree, add things to it in upper levels
    if depth == 0:
      return evaluate(board), [], {}


    # recursive cases: go through each possible move, and min or max depending on the player
    possible_moves = generateMoves(side, board, flags)
    
    value = -math.inf
    moveList = []
    moveTree = {}

    best_move = None # use best_move to find the best move on the current level

    # max_{action}minimax(succ(node, action)) if player == MAX
    # if we are on the max player, we want to maximize the minimax of subsequent possible moves   
    if side == False: # player 0, white, plays next
      value = -math.inf

      for move in possible_moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # from mp documentation
        recValue, recMoveList, recMoveTree = minimax(newside, newboard, newflags, depth - 1) # recursive value on new moves one level

        # update the best move at this level ~ only update moveList and value. 
        if recValue > value:
          value = recValue 
          best_move = move
          moveList = recMoveList
        # since we evaluate all of these possible moves, we add them to the moveTree outside of the if condition
        moveTree[encode(*move)] = recMoveTree # from mp documentation
    
    # min_{action}minmax(succ(node, action)) if player == MIN
    # if we are on the min player's turn, we want to minimize the subequent action they can take
    elif side == True:
      value = math.inf

      for move in possible_moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # from mp documentation
        recValue, recMoveList, recMoveTree = minimax(newside, newboard, newflags, depth - 1)
        if recValue < value:
          value = recValue  
          best_move = move
          moveList = recMoveList
        moveTree[encode(*move)] = recMoveTree # from mp documentation


    # after we have checked all the possible moves, we have to only choose the best move to append to the move list.
    moveList.insert(0, best_move) # have to append to beginning because otherwise you'd be going backwards. use list.insert(0, item)
    
    return value, moveList, moveTree

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    # base case: minimax(node) = utility(node) if the node is terminal
    # at depth 0, return evaluate(board), [], and {} 
    # ==> we return empty at bottom for moveList and moveTree, add things to it in upper levels
    if depth == 0:
      return evaluate(board), [], {}


    # recursive cases: go through each possible move, and min or max depending on the player
    possible_moves = generateMoves(side, board, flags)
    
    value = -math.inf
    moveList = []
    moveTree = {}

    best_move = None # use best_move to find the best move on the current level

    # max_{action}minimax(succ(node, action)) if player == MAX
    # if we are on the max player, we want to maximize the minimax of subsequent possible moves   
    if side == False: # player 0, white, plays next
      value = -math.inf

      for move in possible_moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # from mp documentation
        recValue, recMoveList, recMoveTree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta) # recursive value on new moves one level

        # update the best move at this level ~ only update moveList and value. 
        if recValue > value:
          value = recValue 
          best_move = move
          moveList = recMoveList
        # since we evaluate all of these possible moves, we add them to the moveTree outside of the if condition
        moveTree[encode(*move)] = recMoveTree # from mp documentation

        if value >= beta: # don't evaluate if this stuff doesn't matter
          break
        if value > alpha: # adjust alpha if our current move gives bigger alpha
          alpha = value
    
    # min_{action}minmax(succ(node, action)) if player == MIN
    # if we are on the min player's turn, we want to minimize the subequent action they can take
    elif side == True:
      value = math.inf

      for move in possible_moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2]) # from mp documentation
        recValue, recMoveList, recMoveTree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
        if recValue < value:
          value = recValue  
          best_move = move
          moveList = recMoveList
        moveTree[encode(*move)] = recMoveTree # from mp documentation

        if value <= alpha: # don't evaluate if this stuff doesn't matter
          break
        if value < beta: # adjust beta if other player's current move gives lover beta
          beta = value


    # after we have checked all the possible moves, we have to only choose the best move to append to the move list.
    moveList.insert(0, best_move) # have to append to beginning because otherwise you'd be going backwards. use list.insert(0, item)
    
    return value, moveList, moveTree

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    # helper function - will create a move tree based on a sequential path array
    # for approach, find multiple paths for each initial move. at the end, we put this path back into the move tree 
    # because in this case, everything is going iteratively instead of recursively, we need to do it differently than before
    # path of [8, 7, 6, 5, 4, 3, 2, 1] gives {1: {2: {3: {4: {5: {6: {7: {8: {}}}}}}}}}
    def backtrackPath(mTree, path):
      for i in range(len(path)):
        if i == 0:
          mTree[encode(*path[i])] = {}
        else:
          mTree[encode(*path[i])] = {encode(*path[i-1]): mTree[encode(*path[i-1])]}
      return {encode(*path[-1]) : mTree[encode(*path[-1])]}



    value = float('inf')
    moveList = []
    moveTree = {}

    # for each initial move, we need to find the average of breadth number of random paths out to proper depth
    initial_move_averages = []
    initial_moves = generateMoves(side, board, flags)
    for i_move in initial_moves:
      moveTree[encode(*i_move)] = {} # add the initial move to the moveTree
      initial_move = makeMove(side, board, i_move[0], i_move[1], flags, i_move[2]) # take that move
      
      # we need to evaluate breadth number of paths with a depth of (depth - 1) iteratively
      vals_to_avg = []
      for j in range(breadth): 
        path = []
        curr_depth = 1

        i_side, i_board, i_flags = initial_move
        while (curr_depth <= depth):

          # once we reach a leaf node (we have reached the depth in the path)
          if curr_depth == depth:
            value = evaluate(i_board)
            vals_to_avg.append(value)
            
            child_tree = backtrackPath({}, path)
            moveTree[encode(*i_move)].update(child_tree)
            break
          
          # step: we randomly choose another move using chooser and keep making that move.
          possible_moves = [ mv for mv in generateMoves(i_side, i_board, i_flags) ]
          move = chooser(possible_moves) # next move will always be random
          
          # add random move to path
          path.insert(0, move)
          
          # increment loop - depth increases, and move advances to new move
          curr_depth += 1
          i_side, i_board, i_flags = makeMove(i_side, i_board, move[0], move[1], i_flags, move[2])

          

      # for every initial move, we need a path and also we need the average value of outcomes.  
      path.insert(0, i_move) 
      moveList.append(path) # any path taken after initial move is fair game so we take most recent
      
      avg = 0
      for val in vals_to_avg:
        avg += val
      avg /= len(vals_to_avg)
      initial_move_averages.append(avg)

    # now that we have evaluated all the initial moves to proper depth, just find min of array!

    min_idx = 0
    min = float('inf')
    max_idx = 0
    max = float('-inf')

    for i, val in enumerate(initial_move_averages):
      if val < min:
        min = val
        min_idx = i
      if val > max:
        max = val
        max_idx = i

    return min, moveList[min_idx], moveTree
