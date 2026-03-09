from capture_agents import CaptureAgent
from game import Directions

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='OffensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

class OffensiveAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.start = game_state.get_agent_position(self.index)
        self.boundary = self._compute_boundary_positions(game_state)
        self.boundary_home = min(
            self.boundary,
            key=lambda p: abs(p[1] - (game_state.data.layout.height // 2)))

        self.search_depth = 5
        self.min_max_depth = 3

        self.return_food_threshold = 5
        self.close_quarters_distance = 5

        self.danger_dist = 3
        self.endgame_return_buffer = 35
        self.visible_defenders_indices = []

    def choose_action(self, game_state):
        return self.choose_action_offensive(game_state)

    def choose_action_offensive(self, game_state):

        self.visible_defenders_indices = self.get_visible_opponents(game_state)

        enemies = self.get_opponents(game_state)
        chasing_enemies = any(game_state.get_agent_state(i).scared_timer > 0 for i in enemies)

        def OffensiveEval_general(self, game_state):
            my_state = game_state.get_agent_state(self.index)
            my_pos = game_state.get_agent_position(self.index)
            time_left = getattr(game_state.data, "timeleft", 0)
            carrying = my_state.num_carrying

            score = self.get_score(game_state)
            food = self.get_food(game_state).as_list()
            food_dist_min = min((self.get_maze_distance(my_pos, f) for f in food), default=0)
            food_dist = sum((self.get_maze_distance(my_pos, f) for f in food))
            boundary_dist = min(self.get_maze_distance(my_pos, b) for b in self.boundary) if self.boundary else 0
            endgame_pressure = max(0, self.endgame_return_buffer - time_left)
            food_left = len(self.get_food(game_state).as_list())
            team_dist = self._get_distance_to_closest_teammate(game_state)


            # Weighted sum offensive parameters
            offensivescore = 0.0
            offensivescore += 100 * score                           #score
            offensivescore += 25 * carrying                         # value carrying (future score)
            offensivescore += -5 * food_dist_min                     # move toward food
            offensivescore += -7* food_dist
            offensivescore += -50 * boundary_dist * (carrying > 0)  # when carrying, prefer edging home
            offensivescore += -0.3 * endgame_pressure * boundary_dist
            offensivescore += -20 * food_left
            offensivescore += 28 * team_dist


            return offensivescore

        def OffensiveEval_normal(self, game_state):
            d = 20.0 * self._get_min_distances_to_enemies(game_state)
            return d

        def OffensiveEval_chase(self, game_state):
            d = -25.0 * self._get_min_distances_to_enemies(game_state)
            return d

        ############### THESE ARE THE NORMAL OFFENSIVE MODES OPERATIONS SO WE USE THE OffensiveEval_normal evaluation function here ###############

        def Offensive_Normal_min_max(self, game_state):
            best_action = None
            best_value = float("-inf")

            alpha = float("-inf")
            beta = float("inf")

            actions = game_state.get_legal_actions(self.index)
            actions = [a for a in actions if a != Directions.STOP] or actions

            for action in actions:
                successor = game_state.generate_successor(self.index, action)

                eval_fn = lambda s: OffensiveEval_normal(self, s) + OffensiveEval_general(self, s)
                next_agent = (self.index + 1) % game_state.get_num_agents()
                value = self._alphabeta(successor, 0, next_agent, alpha, beta, eval_fn)

                if value > best_value:
                    best_value = value
                    best_action = action

                alpha = max(alpha, best_value)

            return best_action

        def Offensive_Normal_Search(self, game_state):

            def Offensive_Normal_Search_inner(self, game_state, depth, acc):
                if depth == self.search_depth:
                    return acc

                actions = game_state.get_legal_actions(self.index)
                actions = [a for a in actions if a != Directions.STOP] or actions

                succesor_states = [game_state.generate_successor(self.index, action) for action in actions]
                values = [Offensive_Normal_Search_inner(self, state, depth + 1, acc + OffensiveEval_general(self, state) + OffensiveEval_normal(self, state))
                          for state in succesor_states]

                return max(values)

            actions = game_state.get_legal_actions(self.index)
            actions = [a for a in actions if a != Directions.STOP] or actions

            best_action = max(actions,key=lambda action: Offensive_Normal_Search_inner(self,
                                                                                      game_state.generate_successor(self.index, action),
                                                                                      0, 0))
            return best_action

        ############### THESE ARE THE CHASE OFFENSIVE MODES OPERATIONS (after eating a pellet) SO WE USE THE OffensiveEval_chase EVALUATION FUNCTION HERE ###############

        def Offensive_Chase_min_max(self, game_state):
            best_action = None
            best_value = float("-inf")

            alpha = float("-inf")
            beta = float("inf")

            actions = game_state.get_legal_actions(self.index)
            actions = [a for a in actions if a != Directions.STOP] or actions

            for action in actions:
                successor = game_state.generate_successor(self.index, action)

                eval_fn = lambda s: OffensiveEval_chase(self, s) + OffensiveEval_general(self, s)
                next_agent = (self.index + 1) % game_state.get_num_agents()
                value = self._alphabeta(successor, 0, next_agent, alpha, beta, eval_fn)

                if value > best_value:
                    best_value = value
                    best_action = action

                alpha = max(alpha, best_value)

            return best_action

        def Offensive_Chase_Search(self, game_state):

            def Offensive_Chase_Search_inner(self, game_state, depth, acc):
                if depth >= self.search_depth:
                    return acc

                actions = game_state.get_legal_actions(self.index)
                actions = [a for a in actions if a != Directions.STOP] or actions

                succesor_states = [game_state.generate_successor(self.index, action) for action in actions]
                values = [Offensive_Chase_Search_inner(self, state, depth + 1, acc + OffensiveEval_general(self, state) + OffensiveEval_chase(self, state))
                          for state in succesor_states]

                return max(values)

            actions = game_state.get_legal_actions(self.index)
            actions = [a for a in actions if a != Directions.STOP] or actions

            best_action = max(actions,key=lambda action: Offensive_Chase_Search_inner(self,
                                                                                 game_state.generate_successor(self.index, action),
                                                                                0, 0))
            return best_action


        ######## "Statemachine" voor offensive mode #######
        if chasing_enemies and self.visible_defenders_indices == []:
            return Offensive_Chase_Search(self, game_state)
        elif chasing_enemies and not self.visible_defenders_indices == []:
            return Offensive_Chase_min_max(self, game_state)
        elif not chasing_enemies and self.visible_defenders_indices == []:
            return Offensive_Normal_Search(self, game_state)
        elif not chasing_enemies and not self.visible_defenders_indices == []:
            return Offensive_Normal_min_max(self, game_state)


    ####################################################### HELPER FUNCTIONS ########################################################

    def get_visible_opponents(self, game_state):
        res = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            pos = st.get_position()
            if pos is None:
                continue
            if not st.is_pacman:
                res.append(i)
        return res
    def get_indices_involved_in_close_quarters(self, game_state):
        my_pos = game_state.get_agent_position(self.index)
        involved = [self.index]

        if my_pos is None:
            return involved

        for m in self.get_team(game_state):
            if m == self.index:
                continue
            pos = game_state.get_agent_position(m)
            if pos is None:
                continue
            if self.distancer.get_distance(pos, my_pos) < self.close_quarters_distance:
                involved.append(m)

        for i in self.get_visible_opponents(game_state):
            pos = game_state.get_agent_position(i)
            if pos is None:
                continue
            if self.distancer.get_distance(pos, my_pos) < self.close_quarters_distance:
                involved.append(i)

        return involved
    def _alphabeta(self, state, depth, agent_idx, alpha, beta, eval_function):
        """
        General alpha-beta for multi-agent turn-taking.
        """

        if depth >= self.min_max_depth or state.is_over():
            return eval_function(state)

        agent_state = state.get_agent_state(agent_idx)
        if agent_state is None or agent_state.configuration is None:
            return eval_function(state)

        legal_actions = state.get_legal_actions(agent_idx)
        if not legal_actions:
            return eval_function(state)

        involved = self.get_indices_involved_in_close_quarters(state)

        num_agents = state.get_num_agents()

        next_agent = (agent_idx + 1) % num_agents
        while next_agent not in involved:
            next_agent = (next_agent + 1) % num_agents

        next_depth = depth + 1 if next_agent == self.index else depth

        if agent_idx in self.get_team(state):

            value = float("-inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_idx, action)
                value = max(
                    value,
                    self._alphabeta(successor, next_depth, next_agent, alpha, beta, eval_function)
                )
                if value >= beta:
                    return value  # beta cutoff

                alpha = max(alpha, value)

            return value

        elif agent_idx in self.get_opponents(state):
            value = float("+inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_idx, action)

                value = min(
                    value,
                    self._alphabeta(successor, next_depth, next_agent, alpha, beta, eval_function)
                )
                if value <= alpha:
                    return value  # alpha cutoff
                beta = min(beta, value)

            return value
    def _compute_boundary_positions(self, game_state):
        walls = game_state.get_walls()
        width = game_state.data.layout.width
        height = game_state.data.layout.height

        # Same convention as your defender: boundary x differs for red/blue
        if self.red:
            boundary_x = (width // 2) - 1
        else:
            boundary_x = (width // 2)

        boundary = []
        for y in range(height):
            if not walls[boundary_x][y]:
                boundary.append((boundary_x, y))

        return boundary
    def _get_min_distances_to_enemies(self, game_state):
        agent_distances = game_state.get_agent_distances()
        visible_opponents = self.get_visible_opponents(game_state)
        if visible_opponents:
            return min(self.get_maze_distance(game_state.get_agent_position(self.index), game_state.get_agent_position(i)) for i in visible_opponents)
        if agent_distances is not None:
            return min(agent_distances[i] for i in self.get_opponents(game_state))
        return 0
    def _get_distance_to_closest_teammate(self, game_state):
        my_pos = game_state.get_agent_position(self.index)

        if my_pos is None:
            return 0

        teammate_distances = []

        for teammate in self.get_team(game_state):
            if teammate == self.index:
                continue

            pos = game_state.get_agent_position(teammate)
            if pos is None:
                continue

            teammate_distances.append(self.get_maze_distance(my_pos, pos))

        if teammate_distances:
            return min(teammate_distances)

        return 0


