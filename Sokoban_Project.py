# Do not change the code framework - you could lose your grade
# Project 1 - Q1
import os
import copy
from collections import deque
import numpy as np
import operator
import time
from scipy.spatial import distance


class SokubanSolver1:
    def __init__(self, filename):
        f = open(filename, 'r')
        f = open(filename, 'r')
        rawinput = []
        for line in f.readlines():
            rawinput.append(line.strip())
        
        self.game_map = [' '.join(rawinput[i]) for i in range(len(rawinput))]
        self.game_map = [s.split(' ') for s in self.game_map]
        
        # Save all the positions
        available = np.transpose(np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        p_pos = np.transpose(np.where(np.char.find(self.game_map, 'P') == 0))
        b_pos = np.transpose(np.where(np.char.find(self.game_map, 'B') == 0))
        g_pos = np.transpose(np.where(np.char.find(self.game_map, '*') == 0))

        stack = deque()
        stack.append({'box': b_pos, 'player': p_pos})
        self.all_states = list()
        self.all_states.append((tuple(b_pos.tolist()[0]), tuple(p_pos.tolist()[0])))
        action_list = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        while stack:
            state = stack.popleft()
            for a in action_list:
                if (state['player'] + a).tolist()[0] in available:
                    player_new_position = state['player'] + a
                    if ((state['player'] + a) == state['box']).all():
                        if (state['box'] + a).tolist()[0] in available:
                            box_new_position = state['box'] + a
                        else:
                            continue;            
                    else:
                        box_new_position = state['box']
                    next_move_visited = ((tuple(box_new_position.tolist()[0]), tuple(player_new_position.tolist()[0])))
                    if next_move_visited not in self.all_states:
                        self.all_states.append(next_move_visited)
                        successor = {'box': box_new_position, 'player': player_new_position}
                        stack.append(successor)
                    else:
                        continue
                else:
                    continue
        
        self.utilities = {state:0 for state in self.all_states}
        
    def __loadInput(self, filename):
        f = open(filename, 'r')
        rawinput = []
        for line in f.readlines():
            rawinput.append(line.strip())
        return rawinput
		
    def getting_reward(self, player, box, goal):
        reward = 0
        if (box == goal) and distance.euclidean(player, box) == 1:
            reward = 10000
        else:
            reward = reward - 0.5
        return reward
        
    def all_transition_next(self, state, available_space):
        #available_space: where player and box can move (place != #). Type is list
        #state[0]: box. Type is tuple
        #state[1]: player. Type is tuple
        next_move = list()
        action_list = [(-1,0),(1,0),(0,-1),(0,1)]
        box = state[0]
        player = state[1]

        for a in action_list:
            if list(map(operator.add, player, a)) in available_space:
                player_new_position = tuple(map(operator.add, player, a))
                if tuple(map(operator.add, player, a)) == box:
                    if list(map(operator.add, box, a)) in available_space:
                        box_new_position = tuple(map(operator.add, box, a))
                    else:
                        continue;
                else:
                    box_new_position = tuple(box)
                next_move.append((box_new_position, player_new_position))
        
        return next_move
    
    def value_iteration(self):
        available = np.transpose(np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        g_pos = np.where(np.char.find(self.game_map, '*') == 0)
        g_pos = (g_pos[0][0], g_pos[1][0])
        gamma = 0.95
        max_change = 1e5
        # in value_iteration, state format is ((box_i, box_j),(player_i,player_j))
        # and available_action function is for value iteration
        # action_list = np.array([[-1,0],[1,0],[0,-1],[0,1]])

        print(len(self.all_states))

        while max_change >= 1e-12:
            util_pre = self.utilities.copy()
            max_change = 0 
            
            for state in self.all_states:
                #print('current state: ' + str(state))
                # find available actions
                # If current state is goal state, assign 10 to current state's utility
                # (state[0] == g_pos) and abs(sum(map(operator.sub, state[1], state[0]))) == 1
                next_states = self.all_transition_next(state, available)
                temp_util = 0
                if (state[0] == g_pos) and (distance.euclidean(state[0], state[1]) == 1): #and max(self.utilities.values()) != 10:
                    self.utilities[state] = self.getting_reward(state[1], state[0], g_pos)
                elif next_states:
                    best_util = []
                    #print('possible next states: ' + str(next_states))
                    #print('next states for ' + str(state))
                    for x in next_states:
                        # bellman: p(s'|s,a) * r(s,a,s') + y*U(s')
                        # p(s'|s,a): 0.25
                        # r(s,a,s'): self.getting_reward(player, box, goal)
                        # y: gamma
                        # U(s'): util_pre[x]
                        # whenever it takes move, gives penalty is needed. 
                        temp_util = 0.25 * (self.getting_reward(x[1], x[0], g_pos) + (gamma * util_pre[x]))
                        best_util.append(temp_util)
                    #print(best_util)
                    self.utilities[state] = max(best_util)
                #print(self.utilities[state])
                max_change = max(abs(self.utilities[state] - util_pre[state]), max_change)
        return self.utilities
    
    def policy(self, state):
        available = np.transpose(np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        possible_states = self.all_transition_next(state, available)
        temp_policy = dict()
        for s in possible_states:
            temp_policy[s] = self.utilities[s]
        best_state = [k for k,v in temp_policy.items() if max(temp_policy.values()) == v][0]
        return best_state
    	
    def mdp_solve(self):
    	# Save all the positions
        available = np.transpose(np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        g_pos = np.where(np.char.find(self.game_map, '*') == 0)
        g_pos = (g_pos[0][0], g_pos[1][0])
        
        p_pos = np.where(np.char.find(self.game_map, 'P') == 0)
        p_pos = (p_pos[0][0], p_pos[1][0])
        
        b_pos = np.where(np.char.find(self.game_map, 'B') == 0)
        b_pos = (b_pos[0][0], b_pos[1][0])
        
        # Generate Initial State
        initial_state = {'box': b_pos, 'player': p_pos, 'depth': 0}
        stack = deque()
        stack.append(initial_state)
        
        visited_list = set()
        visited_list.add((b_pos, p_pos))

        if max(self.utilities.values()) <= 0:
            return -1
    	
        while stack:
            state = stack.popleft()
            if (state['box'] == g_pos):
                solution = state['depth']
                return solution
            current_state = (state['box'], state['player'])
            next_state = self.policy(current_state)
            #print(next_state)
            stack.append({'box': next_state[0], 'player': next_state[1], 'depth': state['depth'] + 1})
            #print(stack)
        else:
            solution = -1
            return solution
        return solution
	

    def solve(self, inputFilename): 
        rawinput = self.__loadInput(inputFilename)
        # Implement this
        # Start with processing the input, get the initial state and game map 
        # Then implement a search function
        game_map = [' '.join(rawinput[i]) for i in range(len(rawinput))]
        game_map = [s.split(' ') for s in game_map]
        # Save all the positions
        available = np.transpose(np.where(np.char.find(game_map, '#') != 0)).tolist()
        p_pos = np.transpose(np.where(np.char.find(game_map, 'P') == 0))
        b_pos = np.transpose(np.where(np.char.find(game_map, 'B') == 0))
        g_pos = np.transpose(np.where(np.char.find(game_map, '*') == 0))
        # Generate Initial State
        initial_state = {'box': b_pos, 'player': p_pos, 'depth': 0}
        stack = deque()
        stack.append(initial_state)
        visited_list = set()
        visited_list.add((tuple(b_pos.tolist()[0]), tuple(p_pos.tolist()[0])))
        # Action_list
        # [-1,0]: up
        # [1,0]:  down
        # [0,-1]: left
        # [0,1]:  right
        action_list = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        while stack:
            state = stack.popleft()
            if (state['box'] == g_pos).all():
                solution = state['depth']
                return solution
            for a in action_list:
                #next_move = (state['player'] + a).tolist()[0]
                if (state['player'] + a).tolist()[0] in available:
                    player_new_position = state['player'] + a
                    # When player's next move is at box then we need to check below
                    # 1. can box move with that action?
                    if ((state['player'] + a) == state['box']).all():
                        # When box can be moved to player's action direction
                        if (state['box'] + a).tolist()[0] in available:
                            box_new_position = state['box'] + a
                        else:
                            continue;            
                    # player's next move is not related with pushing
                    else:
                        # Leave it to same value
                        box_new_position = state['box']
                    next_move_visited = ((tuple(box_new_position.tolist()[0]), tuple(player_new_position.tolist()[0])))
                    # Check whether this box and player's current position has been visited before
                    if next_move_visited not in visited_list:
                        visited_list.add(next_move_visited)
                        successor = {'box': box_new_position, 'player': player_new_position, 'depth': state['depth'] + 1}
                        stack.append(successor)
                    else:
                        continue
                else:
                    continue    
        else:
            solution = -1
            return solution
        return solution


if __name__=='__main__':
    for i in range(1,11):
        print("Game Number: ", i)
        bfs_time, mdp_time = [], []
        # if i == 5 or i == 10: pass
        # else: continue
        # if i == 5 or i == 10: continue      

        # for j in range(10):
        test_file_number = i # Change this to use different test files
        filename = 'game%d.txt' % test_file_number
        testfilepath = os.path.join('test','Q1', filename)
        Solver = SokubanSolver1(testfilepath)

        #game_map = Solver.return_gamemap(testfilepath)
        start1 = time.time()
        res = Solver.solve(testfilepath)
        end1 = time.time()
        bfs_time.append(end1-start1)
        
        Solver.value_iteration()
        start = time.time()
        mdp_res = Solver.mdp_solve()
        end = time.time()
        mdp_time.append(end-start)

        ansfilename = 'ans%d.txt' % test_file_number 
        answerfilepath = os.path.join('test', 'Q1', ansfilename)
        f = open(answerfilepath, 'r')
        ans = int(f.readlines()[0].strip())

        #print('Your BFS answer is %d. True answer is %d.' % (res, ans))

        #if res == ans:
        #    print('Answer is correct.')
        #else:
        #    print('Answer is wrong.')
        
        # print('Your MDP answer is %d. True answer is %d.' % (mdp_res, ans))

        # if mdp_res == ans:
        #     print('Answer is correct.')
        # else:
        #     print('Answer is wrong.')

        print('BFS Search took : ' + str(end1-start1))
        print('MDP Search took : ' + str(end-start))
        
        # print('BFS Search took : ', sum(bfs_time)/10)
        # print('MDP Search took : ', sum(mdp_time)/10)

        print()
