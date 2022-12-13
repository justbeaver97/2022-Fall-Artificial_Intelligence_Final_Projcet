import os
import collections
from collections import deque
import numpy as np
import operator
import time
import heapq
from scipy.spatial import distance


class SokubanSolver1:
    def __init__(self, filename):
        f = open(filename, 'r')
        rawinput = []
        for line in f.readlines():
            rawinput.append(line.strip())

        self.game_map = [' '.join(rawinput[i]) for i in range(len(rawinput))]
        self.game_map = [s.split(' ') for s in self.game_map]

        # Save all the positions
        available = np.transpose(
            np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        p_pos = np.transpose(np.where(np.char.find(self.game_map, 'P') == 0))
        b_pos = np.transpose(np.where(np.char.find(self.game_map, 'B') == 0))
        g_pos = np.transpose(np.where(np.char.find(self.game_map, '*') == 0))

        stack = deque()
        stack.append({'box': b_pos, 'player': p_pos})
        self.all_states = list()
        self.all_states.append(
            (tuple(b_pos.tolist()[0]), tuple(p_pos.tolist()[0])))
        action_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        while stack:
            state = stack.popleft()
            for a in action_list:
                if (state['player'] + a).tolist()[0] in available:
                    player_new_position = state['player'] + a
                    if ((state['player'] + a) == state['box']).all():
                        if (state['box'] + a).tolist()[0] in available:
                            box_new_position = state['box'] + a
                        else:
                            continue
                    else:
                        box_new_position = state['box']
                    next_move_visited = (
                        (tuple(box_new_position.tolist()[0]), tuple(player_new_position.tolist()[0])))
                    if next_move_visited not in self.all_states:
                        self.all_states.append(next_move_visited)
                        successor = {'box': box_new_position,
                                     'player': player_new_position}
                        stack.append(successor)
                    else:
                        continue
                else:
                    continue

        self.utilities = {state: 0 for state in self.all_states}

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
        # available_space: where player and box can move (place != #). Type is list
        #state[0]: box. Type is tuple
        #state[1]: player. Type is tuple
        next_move = list()
        action_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        box = state[0]
        player = state[1]

        for a in action_list:
            if list(map(operator.add, player, a)) in available_space:
                player_new_position = tuple(map(operator.add, player, a))
                if tuple(map(operator.add, player, a)) == box:
                    if list(map(operator.add, box, a)) in available_space:
                        box_new_position = tuple(map(operator.add, box, a))
                    else:
                        continue
                else:
                    box_new_position = tuple(box)
                next_move.append((box_new_position, player_new_position))

        return next_move

    def value_iteration(self, number):
        available = np.transpose(
            np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        g_pos = np.where(np.char.find(self.game_map, '*') == 0)
        g_pos = (g_pos[0][0], g_pos[1][0])
        gamma = 0.95
        max_change = 1e5
        # in value_iteration, state format is ((box_i, box_j),(player_i,player_j))
        # and available_action function is for value iteration
        # action_list = np.array([[-1,0],[1,0],[0,-1],[0,1]])

        if number == 0:
            print("Number of all possible states: ", len(self.all_states))

        while max_change >= 1e-12:
            util_pre = self.utilities.copy()
            max_change = 0
            for state in self.all_states:
                # find available actions
                # If current state is goal state, assign 10 to current state's utility
                # (state[0] == g_pos) and abs(sum(map(operator.sub, state[1], state[0]))) == 1
                next_states = self.all_transition_next(state, available)
                temp_util = 0
                # and max(self.utilities.values()) != 10:
                if (state[0] == g_pos) and (distance.euclidean(state[0], state[1]) == 1):
                    self.utilities[state] = self.getting_reward(
                        state[1], state[0], g_pos)
                elif next_states:
                    best_util = []
                    for x in next_states:
                        # bellman: p(s'|s,a) * r(s,a,s') + y*U(s')
                        # p(s'|s,a): 0.25
                        # r(s,a,s'): self.getting_reward(player, box, goal)
                        # y: gamma
                        # U(s'): util_pre[x]
                        # whenever it takes move, gives penalty is needed.
                        temp_util = 0.25 * \
                            (self.getting_reward(
                                x[1], x[0], g_pos) + (gamma * util_pre[x]))
                        best_util.append(temp_util)
                    # print(best_util)
                    self.utilities[state] = max(best_util)
                # print(self.utilities[state])
                max_change = max(
                    abs(self.utilities[state] - util_pre[state]), max_change)
        return self.utilities

    def policy(self, state):
        available = np.transpose(
            np.where(np.char.find(self.game_map, '#') != 0)).tolist()
        possible_states = self.all_transition_next(state, available)
        temp_policy = dict()
        for s in possible_states:
            temp_policy[s] = self.utilities[s]
        best_state = [k for k, v in temp_policy.items() if max(
            temp_policy.values()) == v][0]
        return best_state

    def mdp_solve(self):
        # Save all the positions
        available = np.transpose(
            np.where(np.char.find(self.game_map, '#') != 0)).tolist()
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
            stack.append(
                {'box': next_state[0], 'player': next_state[1], 'depth': state['depth'] + 1})
        else:
            solution = -1
            return solution

    def solve(self, inputFilename):
        rawinput = self.__loadInput(inputFilename)
        game_map = [' '.join(rawinput[i]) for i in range(len(rawinput))]
        game_map = [s.split(' ') for s in game_map]
        available = np.transpose(
            np.where(np.char.find(game_map, '#') != 0)).tolist()
        p_pos = np.transpose(np.where(np.char.find(game_map, 'P') == 0))
        b_pos = np.transpose(np.where(np.char.find(game_map, 'B') == 0))
        g_pos = np.transpose(np.where(np.char.find(game_map, '*') == 0))
        initial_state = {'box': b_pos, 'player': p_pos, 'depth': 0}
        stack = deque()
        stack.append(initial_state)
        visited_list = set()
        visited_list.add((tuple(b_pos.tolist()[0]), tuple(p_pos.tolist()[0])))
        action_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        while stack:
            state = stack.popleft()
            if (state['box'] == g_pos).all():
                solution = state['depth']
                return solution
            for a in action_list:
                if (state['player'] + a).tolist()[0] in available:
                    player_new_position = state['player'] + a
                    if ((state['player'] + a) == state['box']).all():
                        if (state['box'] + a).tolist()[0] in available:
                            box_new_position = state['box'] + a
                        else:
                            continue
                    else:
                        box_new_position = state['box']
                    next_move_visited = (
                        (tuple(box_new_position.tolist()[0]), tuple(player_new_position.tolist()[0])))
                    if next_move_visited not in visited_list:
                        visited_list.add(next_move_visited)
                        successor = {
                            'box': box_new_position, 'player': player_new_position, 'depth': state['depth'] + 1}
                        stack.append(successor)
                    else:
                        continue
                else:
                    continue
        else:
            solution = -1
            return solution


class SokubanSolver2:
    def __loadInput(self, filename):
        f = open(filename, 'r')
        rawinput = []
        for line in f.readlines():
            rawinput.append(line.strip())
        return rawinput

    def preprocess(self, input):
        output = []
        for i in range(len(input)):
            if "B" in input[i]:
                coordinate_B = [i, input[i].find("B")]
            if "P" in input[i]:
                coordinate_P = [i, input[i].find("P")]
            if "*" in input[i]:
                coordinate_star = [i, input[i].find("*")]
            output.append(list(input[i]))
        output[coordinate_B[0]][coordinate_B[1]], output[coordinate_P[0]][coordinate_P[1]
                                                                          ], output[coordinate_star[0]][coordinate_star[1]] = ".", ".", "."
        return output, coordinate_B, coordinate_P, coordinate_star

    def heuristic(self, state, target):
        dist = np.linalg.norm(np.array(state)-np.array(target))
        return dist

    def move_keeper(self, map_array, B, P, star, distance):
        dir, push, cost, num = ([-1, 0], [0, 1], [1, 0], [0, -1]), 0, 0, 0
        f_n = distance + cost
        start = {
            'player': np.array(P),
            'box': np.array(B),
            'push': push,
            'cost': cost,
            'f_n':  distance + cost
        }
        queue = collections.deque()
        queue.append(start)
        heap_queue = [((f_n, num, queue[0]))]
        num += 1
        heapq.heapify(heap_queue)

        visited = set()
        visited.add(((tuple(P), tuple(B))))

        while heap_queue:
            current_node = heapq.heappop(heap_queue)
            for i in range(len(dir)):
                new_x, new_y = current_node[2]['player'][0] + \
                    dir[i][0], current_node[2]['player'][1]+dir[i][1]

                if 0 <= new_x < len(map_array) and 0 <= new_y < len(map_array[0]):
                    if ((tuple([new_x, new_y]), tuple(current_node[2]['box'].tolist()))) not in visited:
                        # if the keeper moves towards coordinate that has box
                        if ([new_x, new_y] == current_node[2]['box'].tolist()):
                            if 0 <= new_x+dir[i][0] < len(map_array) and 0 <= new_y+dir[i][1] < len(map_array[0]):
                                if map_array[new_x+dir[i][0]][new_y+dir[i][1]] != "#":
                                    new_node = {
                                        'player': np.array([new_x, new_y]),
                                        'box': np.array([current_node[2]['box'][0]+dir[i][0], current_node[2]['box'][1]+dir[i][1]]),
                                        'push': current_node[2]['push']+1,
                                        'cost': current_node[2]['cost'],
                                        'f_n':  Solver2.heuristic([new_x, new_y], star) + current_node[2]['cost']
                                    }
                                    heapq.heappush(heap_queue, (Solver2.heuristic(
                                        [new_x, new_y], star) + current_node[2]['cost'], num, new_node))
                                    num += 1
                                    visited.add((tuple([new_x, new_y]), tuple(
                                        [current_node[2]['box'][0]+dir[i][0], current_node[2]['box'][1]+dir[i][1]])))
                                    if [current_node[2]['box'][0]+dir[i][0], current_node[2]['box'][1]+dir[i][1]] == star:
                                        return current_node[2]['push']+1
                        else:
                            if map_array[new_x][new_y] == ".":
                                new_node = {
                                    'player': np.array([new_x, new_y]),
                                    'box': np.array(current_node[2]['box'].tolist()),
                                    'push': current_node[2]['push'],
                                    'cost': current_node[2]['cost']+1,
                                    'f_n':  Solver2.heuristic([new_x, new_y], star) + current_node[2]['cost']
                                }
                                heapq.heappush(heap_queue, (Solver2.heuristic(
                                    [new_x, new_y], star) + current_node[2]['cost'], num, new_node))
                                num += 1
                                visited.add((tuple([new_x, new_y]), tuple(
                                    [current_node[2]['box'][0], current_node[2]['box'][1]])))
        return -1

    def solve(self, inputFilename):
        rawinput = self.__loadInput(inputFilename)
        map_array, B, P, star = Solver2.preprocess(rawinput)
        distance = Solver2.heuristic(B, P)
        solution = Solver2.move_keeper(map_array, B, P, star, distance)
        return solution


if __name__ == '__main__':
    for i in range(1, 11):
        print("Game Number: ", i)
        bfs_time, mdp_time, astar_time, iteration_time = [], [], [], []

        test_file_number = i
        filename = 'game%d.txt' % test_file_number
        testfilepath = os.path.join('testcase', filename)
        Solver = SokubanSolver1(testfilepath)
        Solver2 = SokubanSolver2()

        for j in range(100):
            start_bfs = time.time()
            res = Solver.solve(testfilepath)
            end_bfs = time.time()
            bfs_time.append(end_bfs-start_bfs)

            start_astar = time.time()
            res = Solver2.solve(testfilepath)
            end_astar = time.time()
            astar_time.append(end_astar-start_astar)

            start_iteration = time.time()
            Solver.value_iteration(j)
            end_iteration = time.time()
            iteration_time.append(end_iteration-start_iteration)

            start_mdp = time.time()
            mdp_res = Solver.mdp_solve()
            end_mdp = time.time()
            mdp_time.append(end_mdp-start_mdp)

            ansfilename = 'ans%d.txt' % test_file_number
            answerfilepath = os.path.join('testcase', ansfilename)
            f = open(answerfilepath, 'r')
            ans = int(f.readlines()[0].strip())

            if j == 0:
                print('Your MDP answer is %d. True answer is %d.' %
                      (mdp_res, ans), end=' ')
                if mdp_res == ans:
                    print('Answer is correct.')
                else:
                    print('Answer is wrong.')

        print('BFS Search took : ', sum(bfs_time)/100)
        print('A* search took:   ', sum(astar_time)/100)
        print('Iteration took:   ', sum(iteration_time)/100)
        print('MDP Search took : ', sum(mdp_time)/100)
        print()
