# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################



def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveMinimaxAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.

    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}




MODE_DEFEND = "defend"
MODE_RETURN = "return"
MODE_ATTACK = "attack"

class DefensiveMinimaxAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.prev_def_food = self.get_food_you_are_defending(game_state).as_list()
        self.boundary = self._compute_boundary_positions(game_state)
        # Choose a stable "home" on the boundary.
        h = game_state.data.layout.height
        mid_y = h // 2
        boundary_sorted = sorted(self.boundary, key=lambda p: p[1])
        mid_candidates = sorted(boundary_sorted, key=lambda p: abs(p[1] - mid_y))
        self.boundary_home = mid_candidates[0] if len(mid_candidates) > 0 else None

        # Build a small patrol strip around the middle of the map (3 points if available).
        self.boundary_patrol = mid_candidates[:3] if len(mid_candidates) >= 3 else mid_candidates
        # Used to discourage panicky retreats into the start corner.
        self.start_avoid_weight = 0.5
        self.patrol_i = 0

        # Don't let patrol chase a missing-food event too deep into our territory.
        self.max_patrol_chase_dist = 6

        # Missing-food chase should be short-lived (otherwise we get dragged to corners).
        self.missing_chase_target = None
        self.missing_chase_ttl = 0  # in turns

        self.debug = False

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)

        # mode selection
        invaders = self._visible_invaders(game_state)
        if my_state.is_pacman:
            mode = MODE_RETURN
        else:
            mode = MODE_DEFEND

        # act
        if mode == MODE_RETURN:
            if self.debug:
                print(f"[DEF {self.index}] MODE=RETURN")
            return self._act_return(game_state)

        # DEFEND:
        # If we're scared, do NOT rush an invader; hold the boundary and keep distance.
        if my_state.scared_timer > 0:
            if self.debug:
                print(f"[DEF {self.index}] MODE=SCARED scared_timer={my_state.scared_timer} invaders={len(invaders)}")
            return self._act_scared(game_state, invaders)

        if len(invaders) > 0:
            # only then: minimax
            if self.debug:
                print(f"[DEF {self.index}] MODE=MINIMAX invaders={len(invaders)}")
            return self._minimax_root(game_state, depth=2)

        # no invaders visible -> patrolling 
        if self.debug:
            print(f"[DEF {self.index}] MODE=PATROL chase_ttl={self.missing_chase_ttl} chase_target={self.missing_chase_target}")
        return self._act_patrol(game_state)

    
    # Return mode
    def _act_return(self, game_state):
        actions = list(game_state.get_legal_actions(self.index))
        my_pos = game_state.get_agent_position(self.index)

        # Prefer returning to a central boundary anchor.
        target = self.boundary_home if self.boundary_home is not None else min(self.boundary, key=lambda b: self.get_maze_distance(my_pos, b))
        cur_start_d = self.get_maze_distance(my_pos, self.start)

        best_a, best_score = None, -10**9
        for a in actions:
            s = game_state.generate_successor(self.index, a)
            pos = s.get_agent_position(self.index)

            # Main objective: get back to boundary anchor.
            score = -10 * self.get_maze_distance(pos, target)

            # Avoid drifting into the start corner while doing so.
            start_d = self.get_maze_distance(pos, self.start)
            score += self.start_avoid_weight * (start_d - cur_start_d)

            if a == Directions.STOP:
                score -= 1

            if score > best_score:
                best_score, best_a = score, a

        return best_a if best_a is not None else Directions.STOP

    # Patrol mode
    def _act_patrol(self, game_state):
        # Detect missing defended food .
        now_food = self.get_food_you_are_defending(game_state).as_list()
        missing = list(set(self.prev_def_food) - set(now_food))
        self.prev_def_food = now_food

        my_pos = game_state.get_agent_position(self.index)

        # Refresh a short-lived chase target when food disappears.
        if len(missing) > 0:
            candidate = min(missing, key=lambda p: self.get_maze_distance(my_pos, p))
            # Only chase if it's not too far away from our midline area.
            if self.get_maze_distance(self.boundary_home, candidate) <= self.max_patrol_chase_dist:
                self.missing_chase_target = candidate
                self.missing_chase_ttl = 6  # chase for at most 6 turns

        # If we have an active chase target, pursue it until TTL expires.
        if self.missing_chase_ttl > 0 and self.missing_chase_target is not None:
            target = self.missing_chase_target
            self.missing_chase_ttl -= 1
        else:
            # Otherwise, patrol a small strip on the boundary near the middle.
            if len(self.boundary_patrol) == 0:
                target = self.boundary_home
            else:
                target = self.boundary_patrol[self.patrol_i % len(self.boundary_patrol)]
                if my_pos == target:
                    self.patrol_i += 1

        if my_pos == target:
            return Directions.STOP
        return self._move_towards(game_state, target)

    def _act_scared(self, game_state, invaders):
        my_pos = game_state.get_agent_position(self.index)
        cur_start_d = self.get_maze_distance(my_pos, self.start)
        actions = list(game_state.get_legal_actions(self.index))

        # Avoid crossing into enemy territory while scared.
        safe_actions = []
        for a in actions:
            s = game_state.generate_successor(self.index, a)
            if not s.get_agent_state(self.index).is_pacman:
                safe_actions.append(a)
        if len(safe_actions) > 0:
            actions = safe_actions

        # Choose a boundary anchor to hold.
        if len(self.boundary_patrol) > 0:
            anchor = self.boundary_patrol[self.patrol_i % len(self.boundary_patrol)]
        else:
            anchor = self.boundary_home

        # Precompute closest invader distance in current state.
        inv_pos = [i.get_position() for i in invaders if i.get_position() is not None]

        best_a, best_score = None, -10**9
        for a in actions:
            s = game_state.generate_successor(self.index, a)
            pos = s.get_agent_position(self.index)

            # Base: stay near boundary anchor.
            score = -2 * self.get_maze_distance(pos, anchor)

            # If invaders visible: prefer to keep distance while staying near midline.
            if len(inv_pos) > 0:
                d = min(self.get_maze_distance(pos, p) for p in inv_pos)
                score += 4 * d

            # Discourage moving toward start corner while scared.
            start_d = self.get_maze_distance(pos, self.start)
            score += self.start_avoid_weight * (start_d - cur_start_d)

            # Mild penalty for stopping unless it is genuinely best.
            if a == Directions.STOP:
                score -= 1

            if score > best_score:
                best_score, best_a = score, a

        return best_a if best_a is not None else Directions.STOP

    def _move_towards(self, game_state, target):
        actions = list(game_state.get_legal_actions(self.index))

        # While patrolling/returning, avoid crossing into enemy territory (becoming Pacman)
        # unless STOP is the only option.
        filtered = []
        for a in actions:
            s = game_state.generate_successor(self.index, a)
            st = s.get_agent_state(self.index)
            if not st.is_pacman:
                filtered.append(a)
        if len(filtered) > 0:
            actions = filtered

        best_a, best_d = None, 10**9
        for a in actions:
            s = game_state.generate_successor(self.index, a)
            pos = s.get_agent_position(self.index)
            d = self.get_maze_distance(pos, target)
            # small tie-break: prefer STOP if equal distance (helps holding position)
            if d < best_d or (d == best_d and a == Directions.STOP):
                best_d, best_a = d, a
        return best_a if best_a is not None else Directions.STOP


    # Minimax (alpha-beta)
    def _minimax_root(self, game_state, depth):
        alpha, beta = -10**9, 10**9
        best_val, best_act = -10**9, Directions.STOP
        actions = game_state.get_legal_actions(self.index)

        for a in actions:
            s = game_state.generate_successor(self.index, a)
            val = self._alphabeta(s, depth, self._next_agent(self.index, game_state), alpha, beta)
            if val > best_val:
                best_val, best_act = val, a
            alpha = max(alpha, best_val)
        return best_act

    def _alphabeta(self, state, depth, agent_idx, alpha, beta):
        if depth == 0 or state.is_over():
            return self._eval_defense(state)

        legal = state.get_legal_actions(agent_idx)
        if len(legal) == 0:
            return self._eval_defense(state)

        is_me = (agent_idx == self.index)
        is_enemy = agent_idx in self.get_opponents(state)

        # depth decreases after a full ply
        next_idx = self._next_agent(agent_idx, state)
        next_depth = depth - 1 if is_me else depth

        if is_me:
            v = -10**9
            for a in legal:
                s2 = state.generate_successor(agent_idx, a)
                v = max(v, self._alphabeta(s2, next_depth, next_idx, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v

        if is_enemy:
            v = 10**9
            for a in legal:
                s2 = state.generate_successor(agent_idx, a)
                v = min(v, self._alphabeta(s2, next_depth, next_idx, alpha, beta))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v

        # teammate or unknown: treat as neutral (or max)
        v = 0
        for a in legal:
            s2 = state.generate_successor(agent_idx, a)
            v += self._alphabeta(s2, next_depth, next_idx, alpha, beta)
        return v / float(len(legal))

    def _eval_defense(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # If we're scared, being close to invaders is dangerous; prefer distance and holding the boundary.
        if my_state.scared_timer > 0:
            invaders = self._visible_invaders(game_state)
            score = 0
            if my_state.is_pacman:
                score -= 200
            # Hold boundary while scared
            score -= 2 * min(self.get_maze_distance(my_pos, b) for b in self.boundary)
            if len(invaders) > 0:
                d = min(self.get_maze_distance(my_pos, i.get_position()) for i in invaders)
                score += 4 * d
            # Slightly discourage being near the start corner while scared.
            score += self.start_avoid_weight * self.get_maze_distance(my_pos, self.start)
            return score

        score = 0

        # stay defender
        if my_state.is_pacman:
            score -= 200

        invaders = self._visible_invaders(game_state)
        score -= 1000 * len(invaders)

        if len(invaders) > 0:
            d = min(self.get_maze_distance(my_pos, i.get_position()) for i in invaders)
            score -= 10 * d

        if len(invaders) == 0:
            score -= min(self.get_maze_distance(my_pos, b) for b in self.boundary)

        return score

    # Helpers
    def _visible_invaders(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if e.is_pacman and e.get_position() is not None]

    def _compute_boundary_positions(self, game_state):
        layout = game_state.data.layout
        w, h = layout.width, layout.height
        walls = game_state.get_walls()

        mid_x = (w - 2) // 2
        boundary_x = mid_x if self.red else mid_x + 1

        boundary = []
        for y in range(1, h - 1):
            if not walls[boundary_x][y]:
                boundary.append((boundary_x, y))
        return boundary

    def _next_agent(self, agent_idx, game_state):
        return (agent_idx + 1) % game_state.get_num_agents()