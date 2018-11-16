from __future__ import print_function

import os
import string
import tempfile

import pygraphviz
from networkx.drawing import nx_agraph

JAR_DIR = './baselines/contract'


def regex2dfa(reg_ex, letter='q'):
    transfer_file = tempfile.NamedTemporaryFile(mode='w+')
    command = 'java -jar {}/regex2dfa.jar "{}" {}'.format(
        JAR_DIR, reg_ex, transfer_file.name)
    os.system(command)

    with open(transfer_file.name) as fname:
        dot = fname.read()
        print(dot, file=open('{}.dot'.format(transfer_file.name), 'w'))
    return nx_agraph.from_agraph(pygraphviz.AGraph(dot))


class DFA:
    def __init__(self, reg_ex):
        self.dfa = regex2dfa(reg_ex)
        self.current_state = 'q0'
        self.num_states = len(self.dfa.nodes())
        self.state_ids = dict(zip(self.dfa.nodes(), range(self.num_states)))

    def step(self, action):
        is_accept, self.current_state = self._traverse_dfa(
            action, self.current_state)
        return is_accept

    def reset(self):
        self.current_state = 'q0'

    def _traverse_dfa(self, char, start):
        """
        dfa_dot: dfa in graphviz dot file
        first return value shows if next state is an accept state
        second return value is the next state
        """
        # convert [1-2][0-9] | 3[0-5] to letter in the upper case alph.
        if int(char) >= 10 and int(char) <= 35:
            i = int(char) - 10
            char = '"{}"'.format(string.ascii_uppercase[i])

        dfa = self.dfa
        accept_states = [
            n for n in dfa.nodes()
            if dfa.nodes.data('shape')[n] == 'doublecircle'
        ]
        edges = dfa.edges.data('label')
        transitions = list(filter(lambda x: x[0] == start, edges))
        for transition in transitions:
            if transition[2] == str(char):
                next_state = transition[1]
                if next_state in accept_states:
                    return True, next_state
                else:
                    return False, next_state

        return False, 'q0'

    def states(self):
        return [str(n) for n in self.dfa.nodes()]

    def accepting_states(self):
        return [
            str(n) for n in self.dfa.nodes()
            if self.dfa.nodes.data('shape')[n] == 'doublecircle'
        ]

    def state_id(self):
        return self.state_ids[self.current_state]
