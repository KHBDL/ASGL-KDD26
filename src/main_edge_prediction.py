# -*- coding: utf-8 -*-
"""
Adapted from original implementation by yeonchang:
https://github.com/yeonchang/ASiNE

"""
import os, sys, time, datetime, collections, pickle
import argparse
import json
import random
import tqdm
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy.special import expit
from rdp_accountant import compute_rdp, get_privacy_spent
from sklearn.metrics.pairwise import cosine_similarity
from itertools import repeat


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import generator
import discriminator
import utils
from sklearn.metrics.pairwise import cosine_similarity

class ASGL(object):
    def __init__(self, config):
        print("Start initializing... ")
        self.config = config
        self.pos_graph, self.neg_graph, self.n_node, self.pos_roots, self.neg_roots = \
            utils.read_edges(self.config.train_filename,
                             self.config.test_filename)  # Returns training undirected positive graph, training undirected negative graph, total node count, positive node set, negative node set

        self.train_graph = utils.read_edges_from_file(self.config.train_filename)  # Training graph: [[0,1,1],[0,2,-1]]
        self.test_graph = utils.read_edges_from_file(self.config.test_filename)  # Test graph

        init_delta_D = 0.05
        init_delta_G = 0.05
        # Discriminator weight initialization (can be modified to skip-gram's W, but without corresponding loss and w_in, w_out)
        self.node_emb_init_d = tf.Variable(
            tf.random_uniform([self.n_node, self.config.n_emb], minval=-init_delta_D, maxval=init_delta_D,
                              dtype=tf.float32))
        # Generator weight initialization
        self.node_emb_init_g = tf.Variable(
            tf.random_uniform([self.n_node, self.config.n_emb], minval=-init_delta_G, maxval=init_delta_G,
                              dtype=tf.float32))
        self.task = self.config.task
        # construct or read BFS-trees
        self.pos_partition_trees = {}  # Positive trees
        self.neg_partition_trees = {}  # Negative trees

        self.pos_discriminator = None  # Positive graph discriminator
        self.neg_discriminator = None  # Negative graph discriminator
        self.pos_generator = None      # Positive graph generator
        self.neg_generator = None      # Negative graph generator
        self.build_generator()         # Initialize pos_generator and neg_generator with shared parameters
        self.build_discriminator()     # Initialize pos_discriminator and neg_discriminator with shared parameters

        # Load model checkpoint and initialize TensorFlow session
        self.latest_checkpoint = tf.train.latest_checkpoint(
            self.config.model_log)  # Find latest checkpoint from specified directory for resuming training or loading pretrained model
        self.saver = tf.train.Saver()  # Create Saver object for saving/restoring model variables

        self.config_tf = tf.ConfigProto()  # Create session configuration object (controls GPU memory allocation, parallel threads etc.)
        self.config_tf.gpu_options.allow_growth = True  # Allocate GPU memory on demand
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())  # Create init op for global and local variables
        self.sess = tf.Session(config=self.config_tf)  # Create TensorFlow session with specified config
        self.sess.run(self.init_op)  # Execute variable initialization
        print("ASGL init complete.")


    def build_generator(self):
        # building generator
        with tf.variable_scope("ASGL_generator") as generator_scope:
            self.pos_generator = generator.Generator(n_node=self.n_node,
                                                     node_emb_init=self.node_emb_init_g,
                                                     positive=True, config=self.config)  # positive=True
            generator_scope.reuse_variables()  # Share parameters  *****************
            self.neg_generator = generator.Generator(n_node=self.n_node,
                                                     node_emb_init=self.node_emb_init_g,
                                                     positive=False, config=self.config)  # positive=False

    def build_discriminator(self):
        # building discriminator
        with tf.variable_scope("ASGL_discriminator") as discriminator_scope:
            self.pos_discriminator = discriminator.Discriminator(
                n_node=self.n_node,
                node_emb_init=self.node_emb_init_d,
                positive=True, config=self.config)  # positive=True
            discriminator_scope.reuse_variables()  # Share parameters
            self.neg_discriminator = discriminator.Discriminator(
                n_node=self.n_node,
                node_emb_init=self.node_emb_init_d,
                positive=False, config=self.config)  # positive=False

    def train(
            self):  # This code implements an adversarial learning training process for graph embedding, including bidirectional adversarial training for both positive and negative graphs
        '''The training method mainly consists of:
            1. Model restoration (resuming training from checkpoint)
            2. Positive graph adversarial learning
            3. Negative graph adversarial learning
            4. Periodic model saving
            5. Evaluation and embedding vector output
        '''
        print("Target epsilon:", self.config.epsilon)
        print("Begin training ASGL...")
        checkpoint = tf.train.get_checkpoint_state(self.config.model_log)  # Check if model checkpoint exists
        if checkpoint and checkpoint.model_checkpoint_path and self.config.load_model:  # If config allows loading model, restore model parameters
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        if self.config.partial_node_flag:
            n_node_subsets = self.config.n_node_subsets
        else:
            n_node_subsets = 1
        start_time = time.time()
        ss_times = []
        mean_interval = []
        # orders for RDP
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = np.zeros_like(orders, dtype=float)
        steps = 0
        for epoch in range(
                self.config.n_epochs):  # Train for configured number of epochs, each epoch includes adversarial training for both positive and negative graphs
            print("Epoch {}".format(str(epoch)))
            ss_times.append(time.time())

            # =======Positive graph training process==============
            # Data preparation:
            root_parts = int(len(self.pos_roots) / n_node_subsets)  # pos_roots: positive node set, split into chunks
            np.random.shuffle(self.pos_roots)  # Shuffle positive node set
            pos_roots_parted = list(
                utils.divide_chunks(self.pos_roots,
                                    root_parts))  # Split shuffled node set into chunks of size root_parts

            if epoch > 0 and epoch % self.config.save_steps == 0:  # save the model
                self.saver.save(self.sess, self.config.model_log + ".model.checkpoint")

            print("\t [POS-GRAPH] Adversarial updating...")
            pos_end_flag = False  # Flag indicating whether only partial positive nodes are used for large graphs
            for roots in pos_roots_parted:  # Iterate through positive node chunks
                if self.config.partial_node_flag:
                    pos_end_flag = True

                self.pos_partition_trees = utils.make_bfs_trees(self.pos_graph,
                                                                roots)  # Build tree structure for each chunk: pos_graph: training positive undirected graph; roots: positive node partition set

                # Prepare training data:
                # 1. Data for training pos discriminator: pos_dis_vars: (dis_centers source node set (mixed real/fake), dis_neighbors target node set (mixed real/fake), dis_labels edge labels (real))
                # 2. Data for training pos generator: pos_gen_vars: (gen_pair_node_1 fake source node set, gen_pair_node_2 fake target node set, gen_reward edge labels (discriminator output))
                # Note: real edges are directly connected edges, while fake edges are node pairs on BFS-tree paths with specified window_size (based on sampling probability from generator)
                pos_dis_vars, pos_gen_vars = self._prepare_data_pos(roots)

                # positive discriminator step
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                for d_epoch in range(self.config.n_epochs_dis):
                    # source node set (mixed real/fake), target node set (mixed real/fake), edge labels (real)
                    center_nodes, neighbor_nodes, labels = pos_dis_vars[0], pos_dis_vars[1], pos_dis_vars[2]
                    train_size = len(center_nodes)  # Training set size
                    start_list = list(
                        range(0, train_size, self.config.batch_size_dis))  # Split training set into batches
                    np.random.shuffle(start_list)
                    sampling_prob = self.config.batch_size_dis * 2 / train_size
                    iter = 0
                    for start in start_list:
                        iter += 1
                        if iter > 1:
                            break
                        end = start + self.config.batch_size_dis
                        self.sess.run(self.pos_discriminator.d_updates,
                                      feed_dict={self.pos_discriminator.node_id: np.array(center_nodes[start:end]),
                                                 self.pos_discriminator.node_neighbor_id: np.array(
                                                     neighbor_nodes[start:end]),
                                                 self.pos_discriminator.label: np.array(labels[start:end])})
                        steps += 1
                        rdp = compute_rdp(q=sampling_prob, noise_multiplier=self.config.noise_stddev, steps=steps,
                                          orders=orders)
                        _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=self.config.epsilon)

                # positive generator step
                for g_epoch in range(self.config.n_epochs_gen):
                    # fake source node set, fake target node set, real edge label probability (discriminator output)
                    node_1, node_2, reward = pos_gen_vars[0], pos_gen_vars[1], pos_gen_vars[2]

                    train_size = len(node_1)
                    start_list = list(
                        range(0, train_size, self.config.batch_size_gen))  # Split training set into batches
                    for start in start_list:  # Train pos_generator in batches
                        end = start + self.config.batch_size_gen
                        self.sess.run(self.pos_generator.g_updates,
                                      feed_dict={self.pos_generator.node_id: np.array(node_1[start:end]),
                                                 self.pos_generator.node_neighbor_id: np.array(node_2[start:end]),
                                                 self.pos_generator.reward: np.array(reward[start:end])})
                if pos_end_flag:
                    break
            print(f'Epoch {epoch + 1}: The delta of positive graph is {_delta}, steps is {steps}')

            # =======Negative graph training process==============
            root_parts = int(len(self.neg_roots) / n_node_subsets)  # neg_roots: negative node set, split into chunks
            np.random.shuffle(self.neg_roots)
            neg_roots_parted = list(utils.divide_chunks(self.neg_roots, root_parts))

            print("\t [NEG-GRAPH] Adversarial updating...")

            neg_end_flag = False  # Flag indicating whether only partial negative nodes are used for large graphs
            for roots in neg_roots_parted:  # Iterate through negative node chunks

                if self.config.partial_node_flag:
                    neg_end_flag = True

                self.neg_partition_trees = utils.make_bfs_trees(self.neg_graph,
                                                                roots)  # Build tree structure for each chunk: neg_graph: training negative undirected graph; roots: negative node partition set

                # Prepare training data:
                # 1. Data for training neg/pos discriminator: neg/pos_dis_vars: (dis_centers source node set (mixed real/fake), dis_neighbors target node set (mixed real/fake), dis_labels edge labels (real))
                # 2. Data for training neg/pos generator: neg/pos_gen_vars: (gen_pair_node_1 fake source node set, gen_pair_node_2 fake target node set, gen_reward edge labels (discriminator output))
                # Note: real edges are directly connected edges, while fake edges are node pairs on BFS-tree paths with specified window_size (based on sampling probability from generator)
                neg_dis_vars, neg_gen_vars, pos_dis_vars, pos_gen_vars = self._prepare_neg_data(roots)

                # negative discriminator step
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                # Train neg discriminator
                for d_epoch in range(self.config.n_epochs_dis):
                    center_nodes, neighbor_nodes, labels = neg_dis_vars[0], neg_dis_vars[1], neg_dis_vars[2]

                    train_size = len(center_nodes)
                    start_list = list(range(0, train_size, self.config.batch_size_dis))
                    np.random.shuffle(start_list)
                    iter = 0
                    for start in start_list:
                        iter += 1
                        if iter > 1:
                            break
                        end = start + self.config.batch_size_dis
                        self.sess.run(self.neg_discriminator.d_updates,
                                      feed_dict={self.neg_discriminator.node_id: np.array(center_nodes[start:end]),
                                                 self.neg_discriminator.node_neighbor_id: np.array(
                                                     neighbor_nodes[start:end]),
                                                 self.neg_discriminator.label: np.array(labels[start:end])})
                        sampling_prob = self.config.batch_size_dis * 2 / train_size
                        steps += 1
                        rdp = compute_rdp(q=sampling_prob, noise_multiplier=self.config.noise_stddev, steps=steps,
                                          orders=orders)
                        _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=self.config.epsilon)

                # negative generator step
                for g_epoch in range(self.config.n_epochs_gen):
                    neg_node_1, neg_node_2, neg_reward = neg_gen_vars[0], neg_gen_vars[1], neg_gen_vars[2]

                    train_size = len(neg_node_1)
                    start_list = list(range(0, train_size, self.config.batch_size_gen))
                    np.random.shuffle(start_list)
                    for n_start in start_list:
                        end = n_start + self.config.batch_size_gen
                        self.sess.run(self.neg_generator.g_updates,
                                      feed_dict={self.neg_generator.node_id: np.array(neg_node_1[n_start:end]),
                                                 self.neg_generator.node_neighbor_id: np.array(neg_node_2[n_start:end]),
                                                 self.neg_generator.reward: np.array(neg_reward[n_start:end])})

                # positive discriminator step with fake positive pair from negative generation
                # Train pos discriminator using fake positive node pairs from negative generator
                if self.config.learn_fake_pos == True:
                    center_nodes = []
                    neighbor_nodes = []
                    labels = []
                    dis_all_cnt = 0
                    for d_epoch in range(self.config.n_epochs_dis):
                        center_nodes, neighbor_nodes, labels = pos_dis_vars[0], pos_dis_vars[1], pos_dis_vars[2]

                        train_size = len(center_nodes)
                        start_list = list(range(0, train_size, self.config.batch_size_dis))
                        np.random.shuffle(start_list)
                        iter = 0
                        for start in start_list:
                            iter += 1
                            if iter > 1:
                                break
                            end = start + self.config.batch_size_dis
                            self.sess.run(self.pos_discriminator.d_updates,
                                          feed_dict={self.pos_discriminator.node_id: np.array(center_nodes[start:end]),
                                                     self.pos_discriminator.node_neighbor_id: np.array(
                                                         neighbor_nodes[start:end]),
                                                     self.pos_discriminator.label: np.array(labels[start:end])})
                            sampling_prob = self.config.batch_size_dis * 2 / train_size
                            steps += 1
                            rdp = compute_rdp(q=sampling_prob, noise_multiplier=self.config.noise_stddev, steps=steps,
                                              orders=orders)
                            _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=self.config.epsilon)
                # positive generator step with fake positive pair from negative generation
                # Train pos generator using fake positive node pairs from negative generator
                if self.config.learn_fake_pos == True:
                    for g_epoch in range(self.config.n_epochs_gen):
                        pos_node_1, pos_node_2, pos_reward = pos_gen_vars[0], pos_gen_vars[1], pos_gen_vars[2]

                        train_size = len(pos_node_1)
                        start_list = list(range(0, train_size, self.config.batch_size_gen))
                        np.random.shuffle(start_list)
                        for start in start_list:
                            end = start + self.config.batch_size_gen
                            self.sess.run(self.pos_generator.g_updates,
                                          feed_dict={self.pos_generator.node_id: np.array(pos_node_1[start:end]),
                                                     self.pos_generator.node_neighbor_id: np.array(
                                                         pos_node_2[start:end]),
                                                     self.pos_generator.reward: np.array(pos_reward[start:end])})
                if neg_end_flag:
                    break
            print(f'Epoch {epoch + 1}: The delta of negative graph is {_delta}, steps is {steps}')
            self.evaluation(self, epoch)
            if _delta > self.config.delta:
                self.evaluation(self, epoch)
                break
        self.write_embeddings_to_file()
        print("Training finished")

    def _prepare_data_pos(self, roots):
        dis_centers = []
        dis_neighbors = []
        dis_labels = []

        gen_pair_node_1 = []
        gen_pair_node_2 = []
        gen_paths = []

        def shuffle_and_select(array, k):
            shuffled = random.sample(array, len(array))  # Shuffle the array
            return shuffled[:k]  # Return first k elements

        for i in roots:  # Iterate through positive node list
            if np.random.rand() < self.config.update_ratio:
                real = self.pos_graph[i]  # Get neighbors of node i in positive graph
                real = shuffle_and_select(real, self.config.n_sample_gen)  # Significant impact

                n_sample = self.config.n_sample_gen
                # According to formula 5, return n_sample paths with i as source node
                # Path endpoints (target nodes) stored in fake
                fake, paths_from_i = self.pos_tree_sampler(i, self.pos_partition_trees[i], n_sample,
                                                     for_d=True)
                if paths_from_i is None:
                    continue

                dis_centers.extend([i] * len(real))
                dis_neighbors.extend(real)
                dis_labels.extend([1] * len(real))  # Edge between source i and direct neighbor j labeled 1 (real)

                dis_centers.extend([i] * len(fake))
                dis_neighbors.extend(fake)
                dis_labels.extend([0] * len(fake))  # Edge between source i and BFS tree target j' labeled 0 (fake)

                gen_paths.extend(paths_from_i)  # Store all paths from source node i

        # Store node pair indices from all BFS tree paths
        # Note spacing between node pairs doesn't exceed self.config.window_size
        node_pairs = list(map(utils.extract_pairs_from_path, gen_paths, repeat(self.config.window_size)))
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                gen_pair_node_1.append(pair[0])
                gen_pair_node_2.append(pair[1])

        # Get discriminator results for node pairs in BFS paths
        gen_reward = self.sess.run(self.pos_discriminator.reward,
                                   feed_dict={self.pos_discriminator.node_id: np.array(gen_pair_node_1),
                                              self.pos_discriminator.node_neighbor_id: np.array(
                                                  gen_pair_node_2)})

        return (dis_centers, dis_neighbors, dis_labels), (gen_pair_node_1, gen_pair_node_2, gen_reward)

    def _prepare_neg_data(self, roots):
        neg_dis_centers = []
        neg_dis_neighbors = []
        neg_dis_labels = []

        pos_dis_centers = []
        pos_dis_neighbors = []
        pos_dis_labels = []

        neg_gen_node_pair_1 = []
        neg_gen_node_pair_2 = []

        pos_gen_node_pair_1 = []
        pos_gen_node_pair_2 = []

        gen_paths = []

        def shuffle_and_select(array, k):
            shuffled = random.sample(array, len(array))  # Shuffle the array
            return shuffled[:k]  # Return first k elements

        for i in roots:  # Iterate through each negative source node
            if np.random.rand() < self.config.update_ratio:
                real = self.neg_graph[i]  # Get neighbors of node i in negative graph
                real = shuffle_and_select(real, self.config.n_sample_gen)

                n_sample = self.config.n_sample_gen  # Number of neighbors to sample
                # According to formulas 7 and 8, return n_sample paths with i as source node
                # Path endpoints (negative and positive targets) stored in neg_fakes and pos_fakes
                neg_fakes, pos_fakes, paths_from_i = self.neg_tree_sampler(i, self.neg_partition_trees[i], n_sample,
                                                                     for_d=True)
                if paths_from_i is None:
                    continue

                neg_dis_centers.extend([i] * len(real))
                neg_dis_neighbors.extend(real)
                neg_dis_labels.extend([1] * len(real))  # Real negative edges between i and direct neighbors (label 1)

                neg_dis_centers.extend([i] * len(neg_fakes))
                neg_dis_neighbors.extend(neg_fakes)
                neg_dis_labels.extend(
                    [0] * len(neg_fakes))  # Fake negative edges between i and generated neighbors (label 0)

                if self.config.learn_fake_pos == True:  # If processing fake pos pairs from negative generation
                    if self.pos_graph.get(i) is not None:
                        real = self.pos_graph[i]
                        n_pairs = min(len(real),
                                      len(pos_fakes))  # Take no more than len(pos_fakes) real neighbors for balance

                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(real[:n_pairs])
                        pos_dis_labels.extend(
                            [1] * n_pairs)  # Real positive edges between i and direct neighbors (label 1)

                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(pos_fakes[:n_pairs])
                        pos_dis_labels.extend(
                            [0] * n_pairs)  # Fake positive edges between i and generated neighbors (label 0)

                gen_paths.extend(paths_from_i)  # Store corresponding paths

        # Store node pair indices from all BFS tree paths
        # Note spacing between node pairs doesn't exceed self.config.window_size
        gen_node_pairs = list(map(utils.extract_pairs_from_path, gen_paths, repeat(self.config.window_size)))  # Store node pair indices
        # Define positive/negative nature of node pairs according to balance rule
        gen_node_pairs_sign = list(map(utils.extract_signs_from_path, gen_paths, repeat(self.config.window_size)))

        for i_path in range(len(gen_node_pairs)):  # i-th pair
            for j_pair in range(len(gen_node_pairs[i_path])):
                if gen_node_pairs_sign[i_path][j_pair] == [-1]:
                    neg_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                    neg_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])
                else:
                    if self.config.learn_fake_pos == True:
                        pos_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                        pos_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])

        # Get negative discriminator results for node pairs
        gen_neg_reward = self.sess.run(self.neg_discriminator.reward,
                                       feed_dict={self.neg_discriminator.node_id: np.array(neg_gen_node_pair_1),
                                                  self.neg_discriminator.node_neighbor_id: np.array(
                                                      neg_gen_node_pair_2)})

        # Get positive discriminator results for node pairs
        gen_pos_reward = self.sess.run(self.pos_discriminator.reward,
                                       feed_dict={self.pos_discriminator.node_id: np.array(pos_gen_node_pair_1),
                                                  self.pos_discriminator.node_neighbor_id: np.array(
                                                      pos_gen_node_pair_2)})

        return (neg_dis_centers, neg_dis_neighbors, neg_dis_labels), \
            (neg_gen_node_pair_1, neg_gen_node_pair_2, gen_neg_reward), \
            (pos_dis_centers, pos_dis_neighbors, pos_dis_labels), \
            (pos_gen_node_pair_1, pos_gen_node_pair_2, gen_pos_reward)

    def pos_tree_sampler(self, root, tree, sample_num, for_d):
        """
        Sample nodes from positive BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree structure in format {node: [parent, child1, child2,...]}
            sample_num: number of required samples
            for_d: bool, whether samples are for discriminator (True) or generator (False)
        Returns:
            fakes: list, indices of sampled nodes
            paths: list, paths from root to sampled nodes

        Input: root node of positive BFS tree, tree structure, sample count, and usage flag
        Output: list of sampled nodes and paths from root to each sampled node
        Core function: Samples nodes from BFS tree according to probability distribution for adversarial training
        """
        fakes = []  # Store sampled target nodes
        paths = []  # Store paths
        n = 0
        iter = 0
        while iter < sample_num:  # Continue sampling until enough samples are obtained
            current_node = root  # Start from root node
            previous_node = -1  # Initialize previous node as -1
            paths.append([])  # Add empty path for new sample
            is_root = True  # Flag indicating if at root node
            paths[n].append(current_node)  # Add root node to path

            while True:  # Process node neighbors
                node_neighbor = tree[current_node][1:] if is_root else tree[
                    current_node]  # Skip parent if root node; otherwise get all neighbors
                is_root = False
                if len(node_neighbor) == 0:  # Tree only has root node
                    return None, None

                if for_d:  # Skip 1-hop nodes (positive fakes) when sampling for discriminator
                    if node_neighbor == [root]:
                        return None, None

                    if root in node_neighbor:
                        node_neighbor.remove(root)

                # Get similarity scores between current node and all others using generator (formula 6)
                target_score = self.sess.run(self.pos_generator.target_score,
                                             feed_dict={self.pos_generator.target_node: np.array(
                                                 [current_node])})

                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]  # Extract scores for neighbors
                relevance_probability = np.nan_to_num(relevance_probability)  # Handle NaN values

                relevance_probability = utils.softmax(relevance_probability)  # Apply softmax (formula 5)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[
                    0]  # Randomly select next node based on probability
                paths[n].append(next_node)
                if iter > 1:  # Skip first-order neighbors
                    fakes.append(next_node)

                if len(paths[n]) > self.config.window_size:  # Prevent exceeding path length
                    iter += 1
                    fakes.append(current_node)
                    break

                if next_node == previous_node:  # Terminating condition - reached end
                    iter += 1
                    fakes.append(current_node)
                    break

                previous_node = current_node
                current_node = next_node

            n = n + 1

        return fakes, paths

    def neg_tree_sampler(self, root, tree, sample_num, for_d):
        """
        Sample nodes from negative BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: number of required samples
            for_d: bool, whether samples are for discriminator (True) or generator (False)
        Returns:
            neg_fakes: list, indices of sampled negative nodes
            pos_fakes: list, indices of sampled positive nodes
            paths: list, paths from root to sampled nodes
        """
        neg_fakes = []
        pos_fakes = []
        paths = []
        n = 0
        iter = 0
        while iter < sample_num:  # Continue sampling until enough samples
            current_node = root  # Start from source node
            previous_node = -1
            paths.append([])  # Sample up to sample_num paths
            is_root = True
            paths[n].append(current_node)  # Add source node as path start

            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # Tree only has root
                    return None, None, None
                if for_d:  # Skip 1-hop nodes for discriminator
                    if node_neighbor == [root]:
                        return None, None, None

                    if root in node_neighbor:
                        node_neighbor.remove(root)

                # Get similarity scores using negative generator (formula 6)
                target_score = self.sess.run(self.neg_generator.target_score,
                                             feed_dict={self.neg_generator.target_node: np.array(
                                                 [current_node])})

                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]  # Scores for neighbors
                relevance_probability = np.nan_to_num(relevance_probability)  # Handle NaN
                # Apply balance rule (formula 9)
                relevance_probability = utils.softmax(1 - (utils.softmax(relevance_probability)))

                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[
                    0]  # Select next node based on probability

                while current_node == next_node:  # Avoid duplicates
                    next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]

                paths[n].append(next_node)  # Add neighbor to path
                if iter > 1:  # Skip first-order neighbors
                    if len(paths[n]) % 2 == 1:  # Odd path length - maintain negative correlation
                        pos_fakes.append(next_node)
                    else:  # Even path length
                        neg_fakes.append(next_node)

                if len(paths[n]) > self.config.window_size:
                    iter += 1
                    if len(paths[n]) % 2 == 1:  # Odd path length - negative correlation
                        neg_fakes.append(current_node)
                    else:
                        neg_fakes.append(next_node)

                    if self.config.learn_fake_pos == True:  # Learn fake positive edges from negative generator
                        if len(paths[n]) % 2 == 1:  # Odd path length
                            pos_fakes.append(next_node)
                        else:
                            pos_fakes.append(current_node)
                    break

                if next_node == previous_node:  # Terminating condition - reached end
                    iter += 1
                    if len(paths[n]) % 2 == 1:  # Odd path length - negative correlation
                        neg_fakes.append(current_node)
                    else:
                        neg_fakes.append(next_node)

                    if self.config.learn_fake_pos == True:  # Learn fake positive edges
                        if len(paths[n]) % 2 == 1:
                            pos_fakes.append(next_node)
                        else:
                            pos_fakes.append(current_node)
                    break

                previous_node = current_node
                current_node = next_node

            n = n + 1

        return neg_fakes, pos_fakes, paths


    def write_embeddings_to_file(self):
        """
        Write generator and discriminator embeddings to files
        """
        modes = [self.pos_generator, self.pos_discriminator, self.neg_generator, self.neg_discriminator]

        for i in range(len(modes)):
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(self.config.emb_filenames[i] + ".emb", "w+") as f:
                lines = [str(self.n_node) + "\t" + str(self.config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)


    @staticmethod
    def evaluation(self, epoch):
        emb_sources = [self.pos_generator,
                 self.pos_discriminator]  # Since pos and neg share same gen/dis, only pos needed
        link_method = "concatenation"

        for i in range(len(emb_sources)):  # Evaluate using both pos_gen and pos_dis embeddings
            embedding_matrix = self.sess.run(emb_sources[i].embedding_matrix)
            # Prepare training data
            X_tr = []
            Y_tr = []
            for edge in self.train_graph:  # Process each edge [node_idx, node_idx, edge_weight]
                y = edge[2] if edge[2] == 1 else 0  # Class label: 1 for positive edge, 0 for negative
                emb_1 = np.array(embedding_matrix[edge[0]])
                emb_2 = np.array(embedding_matrix[edge[1]])
                link_emb = utils.aggregate_link_emb(link_method, emb_1, emb_2)  # Combine node embeddings
                X_tr.append(link_emb)
                Y_tr.append(y)

            X_tr = np.array(X_tr)
            Y_tr = np.array(Y_tr)

            # Prepare test data
            X_te = []
            Y_te = []
            for edge in self.test_graph:
                y = edge[2] if edge[2] == 1 else 0
                emb_1 = np.array(embedding_matrix[edge[0]])
                emb_2 = np.array(embedding_matrix[edge[1]])
                link_emb = utils.aggregate_link_emb(link_method, emb_1, emb_2)
                X_te.append(link_emb)
                Y_te.append(y)

            X_te = np.array(X_te)
            Y_te = np.array(Y_te)

            # Train logistic regression classifier
            lr = LogisticRegression(solver='lbfgs', max_iter=10000)
            lr.fit(X_tr, Y_tr)
            prob = lr.predict_proba(X_te)[:, 1]  # Predicted probabilities
            test_y_pred = lr.predict(X_te)  # Predicted labels

            # Calculate evaluation metrics
            auc_s = roc_auc_score(Y_te, prob, average="macro")
            f1_ma = f1_score(Y_te, test_y_pred, average="macro")
            f1_mi = f1_score(Y_te, test_y_pred, average="micro")
            f1_bi = f1_score(Y_te, test_y_pred, average="binary")

            # Format and print results
            result_str = (
                f"Epoch {epoch + 1 } | Mode {emb_sources[i]} → AUC {auc_s:.4f} | "
                f"F1_macro {f1_ma:.4f} | F1_micro {f1_mi:.4f} | F1_bin {f1_bi:.4f}\n"
            )
            print(result_str)

            # Save results to log file
            with open(self.config.result_filename, mode="a+") as f:
                f.write(result_str)


def parse_args():
    # 1. First parse --dataset to know which config to load
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--dataset", nargs="?", default="Bitcoin_Alpha", help="Dataset name.")
    base_args, _ = base_parser.parse_known_args()

    # 2. Load global config file (contains parameters for all datasets)
    config_path = "dataset_configs.json"
    config = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            all_configs = json.load(f)
            config = all_configs.get(base_args.dataset, {})

    # Main argument parser
    parser = argparse.ArgumentParser(description="Run ASiNE.")
    parser.add_argument("--dataset", type=str, default=base_args.dataset, help="Dataset name.")
    parser.add_argument("--n_emb", type=int, default=config.get("n_emb", 5),
                        help="Embedding size. Default is 128.")
    parser.add_argument("--lr", type=float, default=config.get("lr", 0.01),
                        help="Learning rate. Recommended to increase for larger datasets.")
    parser.add_argument("--window_size", type=int, default=config.get("window_size", 3),
                        help="Window size for pair generation. Default is 2.")
    parser.add_argument("--learn_fake_pos", type=bool, default=False,
                        help="Whether to learn fake positive edges from negative generator.")

    parser.add_argument("--n_epochs", type=int, default=config.get("n_epochs", 1000),
                        help="Number of epochs. Default is 70.")
    parser.add_argument("--n_epochs_gen", type=int, default=config.get("n_epochs_gen", 5),
                        help="Number of generator training iterations per epoch.")
    parser.add_argument("--n_epochs_dis", type=int, default=config.get("n_epochs_dis", 5),
                        help="Number of discriminator training iterations per epoch.")
    parser.add_argument("--n_sample_gen", type=int, default=config.get("n_sample_gen", 2),
                        help="Number of samples per generator. Default is 20.")

    parser.add_argument("--batch_size_gen", type=int, default=config.get("batch_size_gen", 128),
                        help="Generator batch size. Default is 64.")
    parser.add_argument("--batch_size_dis", type=int, default=config.get("batch_size_dis", 128),
                        help="Discriminator batch size. Default is 64.")
    parser.add_argument("--n_node_subsets", type=int, default=25,
                        help="Number of subsets for large datasets.")

    parser.add_argument("--lambda_gen", type=float, default=1e-5,
                        help="L2 regularization weight for generator.")
    parser.add_argument("--lambda_dis", type=float, default=1e-5,
                        help="L2 regularization weight for discriminator.")
    parser.add_argument("--noise_stddev", type=float, default=config.get("noise_stddev", "2"),
                        help="Noise stddev for RDP.")
    parser.add_argument("--clip_value", type=float, default=config.get("clip_value", "1.0"),
                        help="Clip value for RDP.")
    parser.add_argument("--epsilon", type=float, default=1,
                        help="Privacy budget.")
    parser.add_argument('--delta', default=10 ** (-5))

    parser.add_argument("--update_ratio", type=int, default=1,
                        help="Update ratio when selecting trees.")
    parser.add_argument("--load_model", type=bool, default=False,
                        help="Whether to load existing model for initialization.")
    parser.add_argument("--skip_model_flag", type=bool, default=False,
                        help="Whether to use skip_model.")
    parser.add_argument("--RDP", type=bool, default=True,
                        help="Whether to use RDP.")
    parser.add_argument("--partial_node_flag", type=bool, default=False,
                        help="Whether to use partial node pairs for large graphs.")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Epoch interval for saving model checkpoints.")

    args = parser.parse_args()

    # Set file paths
    args.train_filename = "../data/" + base_args.dataset + "/" + base_args.dataset + ".train"
    args.test_filename = "../data/" + base_args.dataset + "/" + base_args.dataset + ".test"

    res_fn_path = "../results/" + base_args.dataset + "_dim" + str(args.n_emb) + "_lr" + str(args.lr)
    args.emb_filenames = [res_fn_path + "_gen_p", res_fn_path + "_dis_p",
                          res_fn_path + "_gen_n", res_fn_path + "_dis_n"]
    args.result_filename = res_fn_path + ".results"
    args.modes = ["gen_p", "dis_p", "gen_n", "dis_n"]
    args.model_log = "../log/"
    args.gen_interval = args.n_epochs_gen
    args.dis_interval = args.n_epochs_dis
    args.lr_gen = args.lr
    args.lr_dis = args.lr
    args.task = 'signed_link_prediction'
    print(args)
    print(f"*************Epsilon:{args.epsilon}*****************")
    return args

if __name__ == "__main__":
    config_all = parse_args()
    asgl = ASGL(config_all)
    asgl.train()
