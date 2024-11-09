import random
import codecs
import os
import numpy as np
import argparse

from collections import defaultdict


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions=defaultdict(dict), emissions=defaultdict(dict)):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        # transition file of basename
        with open(f"{basename}.trans", "r") as f:
            for line in f.readlines():
                state0, state1, prob = line.split(" ")
                self.transitions[state0][state1] = float(prob.strip())

        with open(f"{basename}.emit", "r") as f:
            for line in f.readlines():
                state, observation, prob = line.split(" ")
                self.emissions[state][observation] = float(prob.strip())

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        # RESULT STORAGE
        res_state, res_emit = [], []

        # INITIAL STATES & PROBABILITIES of states
        states = [state for state in self.transitions if state != '#']
        init_tran_probs = [prob for prob in self.transitions['#'].values()]
        curr_state = np.random.choice(a=states, p=init_tran_probs)

        # INITIAL STATES & PROBABILITIES of emissions
        emits = [emit for emit in self.emissions[curr_state]]
        init_emit_probs = [prob for prob in self.emissions[curr_state].values()]
        curr_emit = np.random.choice(a=emits, p=init_emit_probs)

        # save initial state and emit
        res_state.append(curr_state)
        res_emit.append(curr_emit)

        while n > 1:
            # GET CORRESPONDING PROBABILITIES base on current state
            curr_state_probs = [prob for prob in self.transitions[curr_state].values()]
            # UPDATE CURRENT STATE: a->input array; p->corresponding weights array
            curr_state = np.random.choice(a=states, p=curr_state_probs)

            # UPDATE CURRENT EMIT base on current state
            curr_emit_candidates = [emit for emit in self.emissions[curr_state]]
            curr_emit_probs = [prob for prob in self.emissions[curr_state].values()]
            curr_emit = np.random.choice(a=curr_emit_candidates, p=curr_emit_probs)

            res_state.append(curr_state)
            res_emit.append(curr_emit)

            n -= 1

        # create a observation instance, and the attribute will be the input for the forward and viterbi
        return Observation(res_state, res_emit)

    def forward(self, observation):
        """
        Viterbi algorithm. Given an Observation (a list of outputs or emissions)
        determine the most likely sequence of states.
        allocate a matrix of s states and n observations.
        :param observation: Given an Observation (a list of outputs or emissions)
        :return: the probability matrix after allocation
        """
        # initialize matrix row: states|col: observations
        states = [s for s in self.transitions if s != '#']
        # +1 for index column
        matrix = [[0.0 for _ in range(len(observation))] for _ in range(len(states))]
        # set hash "#" to 1.0

        # init first column
        for rowNum in range(len(states)):
            # KEY: no emission prob for current <state, observation>
            if observation[0] not in self.emissions[states[rowNum]]:
                matrix[rowNum][0] = 0.0
            else:
                # transfer from # -> current state
                # given current state, the prob of current observation
                matrix[rowNum][0] = (self.emissions[states[rowNum]][observation[0]]
                                     * self.transitions['#'][states[rowNum]] * 1)

        # start to make observations
        for colNum in range(1, len(observation)):        # for each day in the timeline(col)
            for rowNum in range(0, len(states)):         # for each state in this day(row)
                total = 0
                for subRowNum in range(0, len(states)):  # for each probable PREVIOUS state
                    # emission prob: [states[rowNum]][observation[colNum]] current state -> current observation
                    # transition prob: states[subRowNum][states[rowNum]] the probable previous state -> current state
                    # previous state prob: the probable previous state

                    # KEY: no emission prob for current <state, observation>
                    if observation[colNum] not in self.emissions[states[rowNum]]:
                        continue

                    total += (self.emissions[states[rowNum]][observation[colNum]]
                              * self.transitions[states[subRowNum]][states[rowNum]] * matrix[subRowNum][colNum-1])
                # assign the total prob for this <state, observation> cell
                matrix[rowNum][colNum] = total

        return matrix

    def predict_obs_states(self, matrix, final=False):
        """
        predict the state of each observation in the sequence
        :param final: whether we are looking for the state for the final observation
        :param matrix: the matrix after probability allocation by forward
        :return: a sequence of states or the final state
        """
        states = [s for s in self.transitions if s != '#']
        result = []
        for colNum in range(len(matrix[0])):
            obs_states = [(rowNum, rowContent[colNum]) for rowNum, rowContent in enumerate(matrix)]
            obs_row = max(obs_states, key=lambda x: x[1])[0]
            result.append(states[obs_row])

        return [result[-1]] if final else result

    def viterbi(self, observation):
        """
        given an observation, find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        states = [s for s in self.transitions if s != '#']

        # initiate matrix and back_pointers
        matrix = [[0.0 for _ in range(len(observation))] for _ in range(len(states))]
        back_pointers = [[0 for _ in range(len(observation))] for _ in range(len(states))]

        # initiate first column for matrix
        for rowNum in range(len(states)):

            # no emission prob for current <state, observation>
            if observation[0] not in self.emissions[states[rowNum]]:
                matrix[rowNum][0] = 0.0
            else:
                # transfer from # -> current state
                matrix[rowNum][0] = self.emissions[states[rowNum]][observation[0]] * self.transitions['#'][states[rowNum]] * 1

        # initiate first column for back_pointers
        for rowNum in range(len(states)):
            # previous state must be '#', set to -1 as backward ending
            back_pointers[rowNum][0] = -1

        # generate back pointers for each <state, observation>
        for colNum in range(1, len(observation)):
            for rowNum in range(len(states)):

                max_row, max_prob = -1, -1
                for subRowNum in range(len(states)):

                    # no emission prob for current <state, observation>
                    if observation[colNum] not in self.emissions[states[rowNum]]:
                        continue

                    # emission: [states[rowNum]][observation[colNum]] given current state -> current observation
                    # transition: [states[subRowNum]][states[rowNum]] previous state -> current state
                    # previous state prob: matrix[subRowNum][colNum-1]
                    prob = self.emissions[states[rowNum]][observation[colNum]] * self.transitions[states[subRowNum]][states[rowNum]] * matrix[subRowNum][colNum-1]
                    if prob > max_prob:
                        max_prob = round(prob, 9)
                        max_row = subRowNum

                # select highest probability and corresponding stateNum(rowNum) for current <state, observation>
                matrix[rowNum][colNum] = max_prob
                back_pointers[rowNum][colNum] = max_row

        # find start point for going backward, compare with prob of each cell<state, observation>
        curr_row = max([(rowNum, row[-1]) for rowNum, row in enumerate(matrix)], key=lambda x: x[1])[0]
        curr_col = len(observation) - 1

        # update curr_ptr for new state
        curr_ptr = back_pointers[curr_row][curr_col]

        # add start state
        path = [states[curr_row]]
        while curr_col > 0:
            path.append(states[curr_ptr])
            curr_col -= 1
            curr_ptr = back_pointers[curr_ptr][curr_col]

        return path[::-1]


def read_obs(filename):
    """
    helper function to read observation list from obs file
    :param filename: obs filename
    :return: a list of observation sequence
    """
    obs_ls = []
    with open(filename) as f:
        for line in f.readlines():
            if line != '\n':
                obs_ls.append(line.strip().split(" "))
    return obs_ls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HMM arguments parser", epilog='usage examples:\n|'
                                     'python hmm.py partofspeech.browntags.trained --generate 20\n|'
                                     'python hmm.py partofspeech.browntags.trained --forward ambiguous_sents.obs\n|'
                                     'python hmm.py cat --forward my_sample_cat.obs\n|'
                                     'python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs\n|')
    parser.add_argument('file_name', help='training file for hmm to load')
    parser.add_argument('--generate', metavar='<count>', help='generate <n> random emissions', default=None)
    parser.add_argument('--forward', metavar='<filename>',
                        help='generate sequence of states based on a sequence of observations from <filename>', default=None)
    parser.add_argument('--viterbi', metavar='<filename>',
                        help='generate most likely sequence of states base on input observations from <filename>', default=None)
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.file_name)

    if args.generate:
        obs = hmm.generate(int(args.generate))
        print(obs)

    elif args.forward:
        obs_ls = read_obs(filename=args.forward)
        for obs in obs_ls:
            matrix = hmm.forward(obs)
            final_state = hmm.predict_obs_states(matrix, final=True)
            print(f"[Observation Sequence] {' '.join(obs)}")
            print(f"[Most Likely Final State] [{final_state[0]}]")

    elif args.viterbi:
        obs_ls = read_obs(filename=args.viterbi)
        for obs in obs_ls:
            seq = hmm.viterbi(obs)
            print(f"{' '.join(seq)}")
            print(f"{' '.join(obs)}")
