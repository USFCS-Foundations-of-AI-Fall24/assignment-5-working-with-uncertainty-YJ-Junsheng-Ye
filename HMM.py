import argparse
from collections import defaultdict

import numpy as np


# Observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# HMM Model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        """Creates a model from transition and emission probabilities."""
        self.transitions = defaultdict(dict) if transitions is None else transitions
        self.emissions = defaultdict(dict) if emissions is None else emissions

    def load(self, basename):
        """Reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files, including the probabilities."""
        # Load transition probabilities
        with open(f"{basename}.trans", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid line in transitions file: {line}")
                state0, state1, prob = parts
                self.transitions[state0][state1] = float(prob)

            # Handle initial transitions
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.strip().split()
                if parts[0] == '#':
                    if len(parts) != 3:
                        raise ValueError(f"Invalid initial transition line: {line}")
                    _, state1, prob = parts
                    self.transitions['#'][state1] = float(prob)

        # Load emission probabilities
        with open(f"{basename}.emit", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid line in emissions file: {line}")
                state, emission, prob = parts
                self.emissions[state][emission] = float(prob)

    def generate(self, n):
        """Generates an n-length observation by randomly sampling from this HMM."""
        # Result storage
        res_state, res_emit = [], []

        # Initial states & probabilities
        if '#' not in self.transitions:
            raise ValueError("No initial transitions found in the model.")
        initial_transitions = self.transitions['#']
        if not initial_transitions:
            raise ValueError("Initial transitions are empty.")
        states_probs = list(initial_transitions.items())
        states, init_tran_probs = zip(*states_probs)
        init_tran_probs = [float(p) for p in init_tran_probs]
        curr_state = np.random.choice(a=states, p=init_tran_probs)

        # Generate emissions
        while len(res_emit) < n:
            # Emission from current state
            emissions = self.emissions[curr_state]
            emits_probs = list(emissions.items())
            emits, emit_probs = zip(*emits_probs)
            emit_probs = [float(p) for p in emit_probs]
            curr_emit = np.random.choice(a=emits, p=emit_probs)

            # Save state and emission
            res_state.append(curr_state)
            res_emit.append(curr_emit)

            # Transition to next state
            transitions = self.transitions.get(curr_state, {})
            if not transitions:
                # If no transitions from current state, stop generating
                break
            next_states_probs = list(transitions.items())
            next_states, next_probs = zip(*next_states_probs)
            next_probs = [float(p) for p in next_probs]
            curr_state = np.random.choice(a=next_states, p=next_probs)

        return Observation(res_state, res_emit)

    def forward(self, observation):
        """
        Forward algorithm. Given an Observation (a list of outputs or emissions)
        determine the probability matrix.
        :param observation: List of observations
        :return: The forward probability matrix
        """
        states = [s for s in self.transitions if s != '#']
        n_states = len(states)
        n_obs = len(observation)

        # Initialize the forward matrix with zeros
        forward_matrix = np.zeros((n_states, n_obs))

        # Initialize first column
        for i, state in enumerate(states):
            emit_prob = self.emissions[state].get(observation[0], 0.0)
            init_trans_prob = self.transitions['#'].get(state, 0.0)
            forward_matrix[i, 0] = emit_prob * init_trans_prob

        # Iterate over the observations
        for t in range(1, n_obs):
            for j, curr_state in enumerate(states):
                emit_prob = self.emissions[curr_state].get(observation[t], 0.0)
                if emit_prob == 0.0:
                    forward_matrix[j, t] = 0.0
                    continue
                transition_probs = [self.transitions[prev_state].get(curr_state, 0.0) for prev_state in states]
                forward_matrix[j, t] = emit_prob * np.dot(forward_matrix[:, t - 1], transition_probs)

        return forward_matrix

    def predict_obs_states(self, matrix, final=False):
        """
        Predict the state of each observation in the sequence using the forward matrix.
        :param final: Whether to return only the final state.
        :param matrix: The forward probability matrix.
        :return: A list of states or the final state.
        """
        states = [s for s in self.transitions if s != '#']
        if final:
            # Return the state with the highest probability in the last column
            last_probs = matrix[:, -1]
            max_index = np.argmax(last_probs)
            return [states[max_index]]
        else:
            # For each observation, return the state with the highest probability
            predicted_states = [states[np.argmax(matrix[:, t])] for t in range(matrix.shape[1])]
            return predicted_states

    def viterbi(self, observation):
        """
        Viterbi algorithm. Given an Observation, find and return the most likely sequence of states.
        :param observation: List of observations
        :return: Most likely sequence of states
        """
        states = [s for s in self.transitions if s != '#']
        n_states = len(states)
        n_obs = len(observation)

        # Initialize the Viterbi matrix and back pointers
        viterbi_matrix = np.zeros((n_states, n_obs))
        back_pointers = np.zeros((n_states, n_obs), dtype=int)

        # Initialize first column
        for i, state in enumerate(states):
            emit_prob = self.emissions[state].get(observation[0], 0.0)
            init_trans_prob = self.transitions['#'].get(state, 0.0)
            viterbi_matrix[i, 0] = emit_prob * init_trans_prob
            back_pointers[i, 0] = -1  # No predecessor

        # Fill in the Viterbi matrix
        for t in range(1, n_obs):
            for j, curr_state in enumerate(states):
                emit_prob = self.emissions[curr_state].get(observation[t], 0.0)
                max_prob = 0.0
                max_state = 0
                for i, prev_state in enumerate(states):
                    trans_prob = self.transitions[prev_state].get(curr_state, 0.0)
                    prob = viterbi_matrix[i, t - 1] * trans_prob * emit_prob
                    if prob > max_prob:
                        max_prob = prob
                        max_state = i
                viterbi_matrix[j, t] = max_prob
                back_pointers[j, t] = max_state

        # Backtrack to find the most probable state sequence
        state_sequence = []
        last_state = np.argmax(viterbi_matrix[:, -1])
        state_sequence.append(states[last_state])

        for t in range(n_obs - 1, 0, -1):
            last_state = back_pointers[last_state, t]
            state_sequence.append(states[last_state])

        state_sequence.reverse()
        return state_sequence

    def is_safe_spot(self, state):
        """
        Check if the given state is a safe landing spot for the lander.
        :param state: The state to check.
        :return: True if safe, False otherwise.
        """
        safe_spots = {'4,3', '3,4', '4,4', '2,5', '5,5'}
        return state in safe_spots


def read_obs(filename):
    """
    Helper function to read observation lists from an obs file.
    :param filename: obs filename
    :return: A list of observation sequences
    """
    obs_ls = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                obs_ls.append(line.strip().split())
    return obs_ls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="HMM arguments parser",
        epilog='Usage examples:\n|'
               'python hmm.py partofspeech --generate 10\n|'
               'python hmm.py partofspeech --forward ambiguous_sents.obs\n|'
               'python hmm.py cat --forward cat_sequence.obs\n|'
               'python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs\n|')
    parser.add_argument('file_name', help='training file for hmm to load')
    parser.add_argument('--generate', metavar='<count>',
                        help='generate <n> random emissions', default=None)
    parser.add_argument('--forward', metavar='<filename>',
                        help='generate sequence of states based on a sequence of observations from <filename>', default=None)
    parser.add_argument('--viterbi', metavar='<filename>',
                        help='generate most likely sequence of states based on input observations from <filename>', default=None)
    args = parser.parse_args()

    hmm = HMM()
    hmm.load(args.file_name)

    if args.generate:
        obs = hmm.generate(int(args.generate))
        print(' '.join(obs.outputseq))
    elif args.forward:
        obs_ls = read_obs(filename=args.forward)
        for obs in obs_ls:
            matrix = hmm.forward(obs)
            final_state = hmm.predict_obs_states(matrix, final=True)
            print(f"[Observation Sequence] {' '.join(obs)}")
            print(f"[Most Likely Final State] [{final_state[0]}]")
            if 'lander' in args.file_name:
                if hmm.is_safe_spot(final_state[0]):
                    print("Safe to land!")
                else:
                    print("Not safe to land!")
    elif args.viterbi:
        obs_ls = read_obs(filename=args.viterbi)
        for obs in obs_ls:
            seq = hmm.viterbi(obs)
            print(f"{' '.join(seq)}")
            print(f"{' '.join(obs)}")
            if 'lander' in args.file_name:
                if hmm.is_safe_spot(seq[-1]):
                    print("Safe to land!")
                else:
                    print("Not safe to land!")

"""
[Generate list of observations]
python HMM.py cat --generate 10
python HMM.py lander --generate 10
python HMM.py partofspeech --generate 10

[Forward]
python HMM.py cat --forward cat_sequence.obs
python HMM.py lander --forward lander_sequence.obs
python HMM.py partofspeech --forward ambiguous_sents.obs

[Viterbi]
python HMM.py cat --viterbi cat_sequence.obs
python HMM.py lander --viterbi lander_sequence.obs
python HMM.py partofspeech --viterbi ambiguous_sents.obs

"""