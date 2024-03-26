import numpy as np
import sys
import math

class SimpleGibbsSampler:
    def __init__(self, seqs, motif_len, lang, lang_prob, seed):
        self.seqs = seqs
        self.motif_len = motif_len
        self.lang = lang
        self.lang_prob = lang_prob
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.prev_omitted_seq_num = -1
        self.cur_omitted_seq_num = -1
        self.already_found_pos = {} 
        self.cur_starting_pos = [0 for i in range(len(self.seqs))]
        self.num_iterations = 0
        self.prev_highest_window_pos = -1
        #key is seq number 
        #value is position of highest scoring motif



    def run_sampler(self):
        self.num_iterations = self.num_iterations + 1
        print("******************** Iteration " + str(self.num_iterations) + 
                                                    " ********************")
        
        self.cur_starting_pos = self.pick_init_positions()
        #print("starting positions: ")
        #print(self.cur_starting_pos)

        motifs = self.get_msa()
        print("MSA given initialized starting position: ")
        for i in range(len(motifs)):
            print(str(i) + ": " + motifs[i])
        
        # pick a random sequence to omit that wasn't the prev sequence
        while (self.cur_omitted_seq_num == self.prev_omitted_seq_num):
            self.cur_omitted_seq_num = np.random.randint(0, len(self.seqs) - 1)
        self.prev_omitted_seq_num = self.cur_omitted_seq_num
        omitted_seq = self.seqs[self.cur_omitted_seq_num]
        print("Omitted sequence number: " + str(self.cur_omitted_seq_num))
        print("Full sequence: " + omitted_seq)

        count_matrix = self.build_pseudo_count_matrix(motifs)
        #print("psuedo matrix:")
        #print(count_matrix)

        pssm_matrix = self.build_pssm(count_matrix)
        print("PSSM matrix: ")
        print(pssm_matrix)
        
        scores_list = self.score_seq_windows(omitted_seq, pssm_matrix)
        ##print("Scores list:")
        ##print(scores_list)

        max = scores_list[0]
        max_location = 0
        for i in range(len(scores_list)):
            if (max < scores_list[i]):
                max = scores_list[i]
                max_location = i

        print("Highest window score position: " + str(max_location))
        print("Found motif is: " + 
                        omitted_seq[max_location:max_location+self.motif_len])
        print()
        if (self.cur_omitted_seq_num in self.already_found_pos and 
            self.already_found_pos[self.cur_omitted_seq_num] == max_location):
            print("Position " + str(max_location) + " already found in seq " + 
                                            str(self.cur_omitted_seq_num))
            
        else:
            self.already_found_pos[self.cur_omitted_seq_num] = max_location
            self.prev_highest_window_pos = max_location
            print()
            self.run_sampler()
        
        
    # generates random initial positions for the first iteration
    # sets previously omitted sequence's initial position as its formerly
    # highest scoring window
    def pick_init_positions(self):
        result = self.cur_starting_pos
        if (self.num_iterations == 1):
            for i in range(len(self.seqs)):
                result[i] = np.random.randint(0, len(self.seqs[i]) 
                                                            - self.motif_len)
        if (self.prev_highest_window_pos != -1):
            result[self.prev_omitted_seq_num] = self.prev_highest_window_pos

        return result

    # given a list of motifs, counts the amount of each letter in each
    # position with a psuedocount of 1
    def build_pseudo_count_matrix(self, motifs):
        matrix = np.ones((len(self.lang), self.motif_len))
        for i in range(len(motifs)): #iterate over every motif
            if (i != self.cur_omitted_seq_num):
                cur_motif = motifs[i]
                for j in range(self.motif_len): #j represents position
                    cur_letter = cur_motif[j]
                    for k in range(len(self.lang)): #search for letter in lang array
                        if cur_letter == self.lang[k]:
                            matrix[k][j] = matrix[k][j] + 1
                            break
        return matrix

    # given counts of letters from the motifs, generates a log-odds value for
    # each cell, making up the entire PSSM
    def build_pssm(self, count_matrix):
        column_sum = len(self.lang) + len(self.seqs) - 1
        for i in range(len(self.lang)):
            for j in range(self.motif_len):
                count_matrix[i][j] = count_matrix[i][j] / column_sum
                count_matrix[i][j] = count_matrix[i][j] / self.lang_prob
                count_matrix[i][j] = math.log(count_matrix[i][j], 2) 
        return count_matrix
    
    # returns a list of all windows of length motif_len in the given sequence
    # seq based on the PSSM
    def score_seq_windows(self, seq, pssm):
        scores = []
        for i in range(len(seq) - self.motif_len + 1):
            cur_window = seq[i:i+self.motif_len]
            scores.append(self.score_single_window(cur_window, pssm))
        return scores    
    
    # given a string window of length motif_len, returns the score of window
    # based on PSSM
    def score_single_window(self, window, pssm):
        total_score = 0
        for i in range(len(window)):
            cur_letter = window[i]
            for j in range(len(self.lang)):
                if (cur_letter == self.lang[j]):
                    letter = j
            total_score = total_score + pssm[letter][i]
        
        return total_score
    
    #returns a list of motifs based on previously initialized starting positions
    def get_msa(self):
        motifs = []
        for i in range(len(self.seqs)):
            if (self.cur_starting_pos[i] != -1):
                cur_seq = self.seqs[i]
                motifs.append(cur_seq[self.cur_starting_pos[i]:self.cur_starting_pos[i] 
                                                    + self.motif_len])
            
        return motifs
    
    

# given an input stream, reads in a file in FASTA format and returns the two
# starting sequences
def parse_FASTA_file(input):
    sequences = []
    current = ""
    for line in input:
        line = line.strip()
        if line.startswith(">"):
            if current != "":
                sequences.append(current)
            current = ""
        else:
            current += line
    sequences.append(current)
    return sequences

if __name__ == "__main__":
    seqs = parse_FASTA_file(sys.stdin)
    motif_len = 6
    lang = ['A', 'G', 'C', 'T']
    lang_prob = .25
    seed = 20
    sgs = SimpleGibbsSampler(seqs, motif_len, lang, lang_prob, seed)
    msa = sgs.run_sampler()

    