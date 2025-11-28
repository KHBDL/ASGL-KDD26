# -*- coding: utf-8 -*-
"""
Adapted from original implementation by yeonchang:
https://github.com/yeonchang/ASiNE

"""
import os, sys, time, datetime, collections, pickle
import argparse
import json
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import random
import tqdm
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve, auc
from rdp_accountant import compute_rdp
from rdp_accountant import get_privacy_spent

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
        # Discriminator weight initialization (can be modified to skip-gram's W, but lacks corresponding loss and w_in/w_out)
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

        self.pos_discriminator = None  # Discriminator for positive graph
        self.neg_discriminator = None  # Discriminator for negative graph
        self.pos_generator = None  # Generator for positive graph
        self.neg_generator = None  # Generator for negative graph

        # ✅ Reset graph to prevent residual variables from previous runs (especially when changing datasets)
        tf.reset_default_graph()
        self.build_generator()  # Initialize pos_generator and neg_generator with shared parameters
        self.build_discriminator()  # Initialize pos_discriminator and neg_discriminator with shared parameters

        # Load model checkpoint and initialize TensorFlow session
        self.latest_checkpoint = tf.train.latest_checkpoint(
            self.config.model_log)  # Find latest checkpoint from specified directory for resuming training or loading pretrained model
        self.saver = tf.train.Saver()  # Create Saver object for saving/restoring model variables

        self.config_tf = tf.ConfigProto()  # Create session configuration to control behavior (GPU memory allocation, parallel threads etc.)
        self.config_tf.gpu_options.allow_growth = True  # Allocate GPU memory on demand to avoid occupying all memory
        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())  # Create init op for initializing global and local variables
        self.sess = tf.Session(config=self.config_tf)  # Create TensorFlow session with specified config
        self.sess.run(self.init_op)  # Execute initialization op to initialize all variables
        print("End initializing.")

    def build_generator(self):
        # building generator
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as generator_scope:
            self.pos_generator = generator.Generator(n_node=self.n_node,
                                                     node_emb_init=self.node_emb_init_g,
                                                     positive=True, config=self.config)  # positive=True
            generator_scope.reuse_variables()  # Share parameters  *****************
            self.neg_generator = generator.Generator(n_node=self.n_node,
                                                     node_emb_init=self.node_emb_init_g,
                                                     positive=False, config=self.config)  # positive=False

    def build_discriminator(self):
        # building discriminator
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as discriminator_scope:
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
            self):  # This code implements an adversarial learning training process for graph embedding, including bidirectional adversarial training on both positive and negative graphs
        '''The training method mainly consists of the following parts:

            1. Model restoration (resuming training from checkpoint)

            2. Positive graph adversarial learning

            3. Negative graph adversarial learning

            4. Periodic model saving

            5. Evaluation and embedding vector output
        '''
        print("Target epsilon:", self.config.epsilon)
        print("Begin training ASGL...")
        checkpoint = tf.train.get_checkpoint_state(self.config.model_log)  # Check if model checkpoint exists
        if checkpoint and checkpoint.model_checkpoint_path and self.config.load_model:  # If config allows loading model (self.config.load_model is True), restore model parameters
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        if self.config.partial_node_flag:
            n_node_subsets = self.config.n_node_subsets
        else:
            n_node_subsets = 1
        start_time = time.time()
        ss_times = []
        mean_interval = []
        # orders for RDP (Rényi Differential Privacy)
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = np.zeros_like(orders, dtype=float)
        steps = 0
        for epoch in range(
                self.config.n_epochs):  # Train for configured number of epochs, each epoch includes adversarial training on both positive and negative graphs
            print("Epoch {}".format(str(epoch)))
            ss_times.append(time.time())

            # =======Positive Graph Training Process==============
            # Data preparation:
            root_parts = int(
                len(self.pos_roots) / n_node_subsets)  # pos_roots: positive node set, partition positive source nodes
            np.random.shuffle(self.pos_roots)  # Shuffle positive node set
            pos_roots_parted = list(
                utils.divide_chunks(self.pos_roots,
                                    root_parts))  # Divide shuffled node set into chunks of size root_parts

            if epoch > 0 and epoch % self.config.save_steps == 0:  # Save the model periodically
                self.saver.save(self.sess, self.config.model_log + ".model.checkpoint")

            print("\t [POS-GRAPH] Adversarial updating...")
            pos_end_flag = False  # Flag indicating whether only partial positive nodes are used for large graphs
            for roots in pos_roots_parted:  # Iterate through positive node partitions
                if self.config.partial_node_flag:
                    pos_end_flag = True

                self.pos_partition_trees = utils.make_bfs_trees(self.pos_graph,
                                                                roots)  # Build tree structure for each partition: pos_graph: training positive undirected graph; roots: positive node partition set

                # Prepare training data:
                # 1. Data for training pos discriminator: pos_dis_vars: (dis_centers source node set (mixed real/fake), dis_neighbors target node set (mixed real/fake), dis_labels edge labels (real))
                # 2. Data for training pos generator: pos_gen_vars: (gen_pair_node_1 fake source node set, gen_pair_node_2 fake target node set, gen_reward edge labels (discriminator output))
                # Note: "real" edges are directly connected edges, while "fake" edges are node pairs on BFS-tree paths with specified window_size (based on sampling probability from generator)
                pos_dis_vars, pos_gen_vars = self._prepare_pos_data(roots)

                # positive discriminator training step
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                for d_epoch in range(self.config.n_epochs_dis):
                    # Source node set (mixed real/fake), target node set (mixed real/fake), edge labels (real)
                    center_nodes, neighbor_nodes, labels = pos_dis_vars[0], pos_dis_vars[1], pos_dis_vars[2]
                    train_size = len(center_nodes)  # Training set size
                    start_list = list(range(0, train_size, self.config.batch_size_dis))  # Create batch indices
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
                        # steps = (epoch*(self.config.n_epochs_dis)*2) + (d_epoch+1)
                        rdp = compute_rdp(q=sampling_prob, noise_multiplier=self.config.noise_stddev, steps=steps,
                                          orders=orders)
                        _eps, _delta, _ = get_privacy_spent(orders, rdp, target_eps=self.config.epsilon)

                # positive generator training step
                for g_epoch in range(self.config.n_epochs_gen):
                    # fake source node set, fake target node set, real edge label probabilities (discriminator output)
                    node_1, node_2, reward = pos_gen_vars[0], pos_gen_vars[1], pos_gen_vars[2]

                    train_size = len(node_1)
                    start_list = list(range(0, train_size, self.config.batch_size_gen))  # Create batch indices
                    # np.random.shuffle(start_list)
                    for start in start_list:  # Train pos_generator in batches
                        end = start + self.config.batch_size_gen
                        self.sess.run(self.pos_generator.g_updates,
                                      feed_dict={self.pos_generator.node_id: np.array(node_1[start:end]),
                                                 self.pos_generator.node_neighbor_id: np.array(node_2[start:end]),
                                                 self.pos_generator.reward: np.array(reward[start:end])})
                if pos_end_flag:
                    break
            print(f'Epoch {epoch + 1}: The delta of positive graph is {_delta}, steps is {steps}')
            # print(f'Epoch {epoch+1}: The epsilon of positive graph is {_eps}')
            # =======Negative Graph Training Process==============
            root_parts = int(
                len(self.neg_roots) / self.config.n_node_subsets)  # neg_roots: negative node set, partition negative source nodes
            np.random.shuffle(self.neg_roots)
            neg_roots_parted = list(utils.divide_chunks(self.neg_roots, root_parts))

            print("\t [NEG-GRAPH] Adversarial updating...")
            neg_end_flag = False  # Flag indicating whether only partial negative nodes are used for large graphs
            for roots in neg_roots_parted:  # Iterate through negative node partitions

                if self.config.partial_node_flag:
                    neg_end_flag = True

                self.neg_partition_trees = utils.make_bfs_trees(self.neg_graph,
                                                                roots)  # Build tree structure for each partition: neg_graph: training negative undirected graph; roots: negative node partition set

                # Prepare training data:
                # 1. Data for training neg/pos discriminator: neg/pos_dis_vars: (dis_centers source node set (mixed real/fake), dis_neighbors target node set (mixed real/fake), dis_labels edge labels (real))
                # 2. Data for training neg/pos generator: neg/pos_gen_vars: (gen_pair_node_1 fake source node set, gen_pair_node_2 fake target node set, gen_reward edge labels (discriminator output))
                # Note: "real" edges are directly connected edges, while "fake" edges are node pairs on BFS-tree paths with specified window_size (based on sampling probability from generator)
                neg_dis_vars, neg_gen_vars, pos_dis_vars, pos_gen_vars = self._prepare_data_negative(roots)

                # negative discriminator training step
                center_nodes = []
                neighbor_nodes = []
                labels = []
                dis_all_cnt = 0
                # Train negative discriminator
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

                # negative generator training step
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

                # positive discriminator training step using fake positive pairs from negative generation
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
                # positive generator training step using fake positive pairs from negative generation
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

            # print(f'Epoch {epoch+1}: The epsilon of negative graph is {_eps}')
            self.evaluation(self, epoch)
            if _delta > self.config.delta:  # Stop training if privacy budget is exhausted
                break
        self.write_embeddings_to_file()  # Save final embeddings
        print("Complete training")

    def  _prepare_pos_data(self, roots):
        dis_centers = []  # Store source nodes for discriminator training
        dis_neighbors = []  # Store neighbor nodes for discriminator training
        dis_labels = []  # Store edge labels (1=real, 0=fake) for discriminator training

        gen_pair_node_1 = []  # Store first nodes in pairs for generator training
        gen_pair_node_2 = []  # Store second nodes in pairs for generator training
        gen_paths = []  # Store all BFS paths for generator training

        def shuffle_and_select(array, k):
            shuffled = random.sample(array, len(array))  # Shuffle the array
            return shuffled[:k]  # Return first k elements

        for i in roots:  # Iterate through all positive root nodes
            if np.random.rand() < self.config.update_ratio:  # Randomly select nodes based on update ratio
                real = self.pos_graph[i]  # Get real neighbors of node i in positive graph
                real = shuffle_and_select(real, self.config.n_sample_gen)  # Shuffle and select subset of neighbors

                n_sample = self.config.n_sample_gen
                # Sample fake neighbors using BFS tree paths (Equation 5 in paper)
                fake, paths_from_i = self.pos_tree_sampler(i, self.pos_partition_trees[i], n_sample, for_d=True)
                if paths_from_i is None:
                    continue

                # Add real edges to discriminator training data (label=1)
                dis_centers.extend([i] * len(real))
                dis_neighbors.extend(real)
                dis_labels.extend([1] * len(real))  # Label 1 indicates real edges

                # Add fake edges to discriminator training data (label=0)
                dis_centers.extend([i] * len(fake))
                dis_neighbors.extend(fake)
                dis_labels.extend([0] * len(fake))  # Label 0 indicates fake edges

                gen_paths.extend(paths_from_i)  # Store all paths for generator training

        # Extract node pairs from all BFS paths where distance <= window_size
        node_pairs = list(map(utils.extract_pairs_from_path, gen_paths, repeat(self.config.window_size)))
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                gen_pair_node_1.append(pair[0])
                gen_pair_node_2.append(pair[1])

        # Get discriminator's reward (probability) for generator node pairs
        gen_reward = self.sess.run(self.pos_discriminator.reward,
                                   feed_dict={self.pos_discriminator.node_id: np.array(gen_pair_node_1),
                                              self.pos_discriminator.node_neighbor_id: np.array(gen_pair_node_2)})

        return (dis_centers, dis_neighbors, dis_labels), (gen_pair_node_1, gen_pair_node_2, gen_reward)

    def _prepare_data_negative(self, roots):
        neg_dis_centers = []  # Source nodes for negative discriminator
        neg_dis_neighbors = []  # Neighbor nodes for negative discriminator
        neg_dis_labels = []  # Labels for negative discriminator (1=real, 0=fake)

        pos_dis_centers = []  # Source nodes for positive discriminator
        pos_dis_neighbors = []  # Neighbor nodes for positive discriminator
        pos_dis_labels = []  # Labels for positive discriminator (1=real, 0=fake)

        neg_gen_node_pair_1 = []  # First nodes in negative generator pairs
        neg_gen_node_pair_2 = []  # Second nodes in negative generator pairs

        pos_gen_node_pair_1 = []  # First nodes in positive generator pairs
        pos_gen_node_pair_2 = []  # Second nodes in positive generator pairs

        gen_paths = []  # Store all BFS paths

        def shuffle_and_select(array, k):
            shuffled = random.sample(array, len(array))  # Shuffle the array
            return shuffled[:k]  # Return first k elements

        for i in roots:  # Iterate through all negative root nodes
            if np.random.rand() < self.config.update_ratio:  # Randomly select nodes based on update ratio
                real = self.neg_graph[i]  # Get real neighbors of node i in negative graph
                real = shuffle_and_select(real, self.config.n_sample_gen)

                n_sample = self.config.n_sample_gen
                # Sample fake negative and positive neighbors using BFS tree paths (Equations 7-8 in paper)
                neg_fakes, pos_fakes, paths_from_i = self.neg_tree_sampler(i, self.neg_partition_trees[i], n_sample,
                                                                     for_d=True)
                if paths_from_i is None:
                    continue

                # Add real negative edges to discriminator training data (label=1)
                neg_dis_centers.extend([i] * len(real))
                neg_dis_neighbors.extend(real)
                neg_dis_labels.extend([1] * len(real))  # Label 1 indicates real negative edges

                # Add fake negative edges to discriminator training data (label=0)
                neg_dis_centers.extend([i] * len(neg_fakes))
                neg_dis_neighbors.extend(neg_fakes)
                neg_dis_labels.extend([0] * len(neg_fakes))  # Label 0 indicates fake negative edges

                if self.config.learn_fake_pos == True:  # If using fake positive pairs from negative generation
                    if self.pos_graph.get(i) is not None:
                        real = self.pos_graph[i]  # Get real positive neighbors
                        n_pairs = min(len(real), len(pos_fakes))  # Balance dataset size

                        # Add real positive edges to discriminator training data (label=1)
                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(real[:n_pairs])
                        pos_dis_labels.extend([1] * n_pairs)

                        # Add fake positive edges to discriminator training data (label=0)
                        pos_dis_centers.extend([i] * n_pairs)
                        pos_dis_neighbors.extend(pos_fakes[:n_pairs])
                        pos_dis_labels.extend([0] * n_pairs)

                gen_paths.extend(paths_from_i)  # Store paths for generator training

        # Extract node pairs and their signs (positive/negative) from all BFS paths
        gen_node_pairs =  list(map(utils.extract_pairs_from_path, gen_paths, repeat(self.config.window_size)))  # Get node pairs
        gen_node_pairs_sign =list(map(utils.extract_signs_from_path, gen_paths, repeat(self.config.window_size)))  # Get pair signs (-1=negative, 1=positive)

        for i_path in range(len(gen_node_pairs)):  # For each path
            for j_pair in range(len(gen_node_pairs[i_path])):  # For each node pair in path
                if gen_node_pairs_sign[i_path][j_pair] == [-1]:  # Negative edge
                    neg_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                    neg_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])
                else:  # Positive edge
                    if self.config.learn_fake_pos == True:
                        pos_gen_node_pair_1.append(gen_node_pairs[i_path][j_pair][0])
                        pos_gen_node_pair_2.append(gen_node_pairs[i_path][j_pair][1])

        # Get discriminator rewards for negative and positive generator pairs
        gen_neg_reward = self.sess.run(self.neg_discriminator.reward,
                                       feed_dict={self.neg_discriminator.node_id: np.array(neg_gen_node_pair_1),
                                                  self.neg_discriminator.node_neighbor_id: np.array(
                                                      neg_gen_node_pair_2)})

        gen_pos_reward = self.sess.run(self.pos_discriminator.reward,
                                       feed_dict={self.pos_discriminator.node_id: np.array(pos_gen_node_pair_1),
                                                  self.pos_discriminator.node_neighbor_id: np.array(
                                                      pos_gen_node_pair_2)})

        return (neg_dis_centers, neg_dis_neighbors, neg_dis_labels), \
            (neg_gen_node_pair_1, neg_gen_node_pair_2, gen_neg_reward), \
            (pos_dis_centers, pos_dis_neighbors, pos_dis_labels), \
            (pos_gen_node_pair_1, pos_gen_node_pair_2, gen_pos_reward)

    def  pos_tree_sampler(self, root, tree, sample_num, for_d):
        """
        Sample nodes from positive BFS-tree using random walk with generator-guided probabilities
        config:
            root: int, root node to start sampling from
            tree: dict, BFS-tree structure in format {node: [parent, child1, child2,...]}
            sample_num: number of samples required
            for_d: bool, whether samples are for discriminator (True) or generator (False)
        Returns:
            fakes: list, indices of sampled nodes
            paths: list, paths from root to each sampled node
        """
        fakes = []  # Store sampled target nodes
        paths = []  # Store paths from root to sampled nodes
        n = 0
        iter = 0
        while iter < sample_num:  # Continue until we get enough samples
            current_node = root  # Start from root node
            previous_node = -1  # Initialize previous node
            paths.append([])  # Add empty path for new sample
            is_root = True  # Flag for root node
            paths[n].append(current_node)  # Add root to path

            while True:  # Node neighbor processing
                # Get neighbors (skip parent if root)
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # Tree only has root
                    return None, None

                if for_d:  # For discriminator, skip 1-hop neighbors (direct connections)
                    if node_neighbor == [root]:
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)

                # Get generator's similarity scores for current node (Equation 6)
                target_score = self.sess.run(self.pos_generator.target_score,
                                             feed_dict={self.pos_generator.target_node: np.array([current_node])})
                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]  # Scores for neighbors
                relevance_probability = np.nan_to_num(relevance_probability)  # Handle NaN
                relevance_probability = utils.softmax(relevance_probability)  # Softmax (Equation 5)

                # Sample next node based on probability distribution
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]
                paths[n].append(next_node)

                if iter > 1:  # Skip first-order neighbors
                    fakes.append(next_node)

                if len(paths[n]) > self.config.window_size:  # Prevent path from exceeding window size
                    iter += 1
                    fakes.append(current_node)
                    break

                if next_node == previous_node:  # Termination condition - reached end
                    iter += 1
                    fakes.append(current_node)
                    break

                previous_node = current_node
                current_node = next_node

            n = n + 1

        return fakes, paths

    def neg_tree_sampler(self, root, tree, sample_num, for_d):
        """
        Sample nodes from negative BFS-tree using modified random walk with generator probabilities
        config:
            root: int, root node to start sampling from
            tree: dict, BFS-tree structure
            sample_num: number of samples required
            for_d: bool, whether samples are for discriminator (True) or generator (False)
        returns:
            neg_fakes: list, sampled nodes with negative relations
            pos_fakes: list, sampled nodes with positive relations (from balance theory)
            paths: list, paths from root to sampled nodes
        """
        neg_fakes = []  # Nodes with negative relations
        pos_fakes = []  # Nodes with positive relations (from balance theory)
        paths = []  # Store paths
        n = 0
        iter = 0
        while iter < sample_num:  # Continue until enough samples
            current_node = root  # Start from root
            previous_node = -1
            paths.append([])  # New path
            is_root = True
            paths[n].append(current_node)  # Add root

            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # Only root exists
                    return None, None, None

                if for_d:  # Skip 1-hop neighbors for discriminator
                    if node_neighbor == [root]:
                        return None, None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)

                # Get generator's similarity scores (Equation 6)
                target_score = self.sess.run(self.neg_generator.target_score,
                                             feed_dict={self.neg_generator.target_node: np.array([current_node])})
                target_score.reshape(target_score.shape[-1])
                relevance_probability = target_score[0, node_neighbor]
                relevance_probability = np.nan_to_num(relevance_probability)
                # Modified probability using balance theory (Equation 9)
                relevance_probability = utils.softmax(1 - (utils.softmax(relevance_probability)))

                # Sample next node
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]
                while current_node == next_node:  # Avoid self-loops
                    next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]

                paths[n].append(next_node)  # Add to path

                if iter > 1:  # Skip first-order neighbors
                    # Apply balance theory - alternate positive/negative based on path length
                    if len(paths[n]) % 2 == 1:  # Odd path length -> positive
                        pos_fakes.append(next_node)
                    else:  # Even path length -> negative
                        neg_fakes.append(next_node)

                if len(paths[n]) > self.config.window_size:  # Window size limit
                    iter += 1
                    if len(paths[n]) % 2 == 1:
                        neg_fakes.append(current_node)
                    else:
                        neg_fakes.append(next_node)

                    if self.config.learn_fake_pos == True:  # Optional positive sampling
                        if len(paths[n]) % 2 == 1:
                            pos_fakes.append(next_node)
                        else:
                            pos_fakes.append(current_node)
                    break

                if next_node == previous_node:  # Termination
                    iter += 1
                    if len(paths[n]) % 2 == 1:
                        neg_fakes.append(current_node)
                    else:
                        neg_fakes.append(next_node)

                    if self.config.learn_fake_pos == True:
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
        Save embedding matrices of all models (generators/discriminators) to files
        """
        modes = [self.pos_generator, self.pos_discriminator, self.neg_generator, self.neg_discriminator]

        for i in range(len(modes)):
            # Get embeddings and prepend node indices
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])

            # Format embeddings as strings
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]

            # Write to file with header
            with open(self.config.emb_filenames[i] + ".emb", "w+") as f:
                lines = [str(self.n_node) + "\t" + str(self.config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)

    @staticmethod
    def evaluation(self, epoch):
        """
        Evaluate model performance on link prediction or node clustering tasks
        Args:
            epoch: current epoch number for logging
        """
        # Use pos_generator and pos_discriminator since they share parameters with negative versions
        modes = [self.pos_generator, self.pos_discriminator]
        link_method = "concatenation"  # Method to combine node embeddings for link prediction

        for i in range(len(modes)):  # Evaluate both generator and discriminator embeddings
            embedding_matrix = self.sess.run(modes[i].embedding_matrix)
            print('Conduct the node_clustering task')
            # Calculate cosine similarity between all node embeddings
            cos_sim_emb = cosine_similarity(np.array(embedding_matrix))
            cos_sim_positive = 0  # Accumulator for positive edge similarities
            cos_sim_negative = 0  # Accumulator for negative edge similarities
            count_positive = 0  # Counter for positive edges
            count_negative = 0  # Counter for negative edges

            # Calculate average similarity for positive and negative edges
            for edge in self.test_graph:
                if edge[2] == 1:  # Positive edge
                    count_positive += 1
                    cos_sim_positive += cos_sim_emb[edge[0]][edge[1]]
                else:  # Negative edge
                    count_negative += 1
                    cos_sim_negative += cos_sim_emb[edge[0]][edge[1]]

            # Calculate final scores
            score_postive = (cos_sim_positive / count_positive)  # Avg similarity for positive edges
            score_negtive = (cos_sim_negative / count_negative)  # Avg similarity for negative edges
            # Structural Similarity Index (SSI) score
            SSI_score = 1 / (abs(score_postive - 1) + abs(score_negtive + 1))

            # Format and print results
            result_str = (
                f"Epoch {epoch + 1} | Mode {modes[i]} → SSI_score {SSI_score:.4f} \n"
            )
            print(result_str)


def parse_args():
    """
    Parse command line arguments and load configuration from JSON file
    Returns:
        args: Namespace object containing all configuration parameters
    """
    # 1. First parse just the dataset argument to know which config to load
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--dataset", nargs="?", default="Bitcoin_Alpha", help="Dataset name (wikirfa, Bitcoin_Alpha, Bitcoin_OTC, Epinions, Slashdot, Epinions_large)")

    base_args, _ = base_parser.parse_known_args()  # Only parse --dataset initially

    # 2. Load configuration from JSON file
    config_path = "dataset_configs_clustering.json"  # Configuration file path
    config = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            all_configs = json.load(f)
            # Get config for specified dataset
            config = all_configs.get(base_args.dataset, {})

    # 3. Define full argument parser with defaults from config file
    parser = argparse.ArgumentParser(description="Run ASiNE.")
    parser.add_argument("--dataset", type=str, default=base_args.dataset, help="Dataset name.")

    # Model hyperparameters
    parser.add_argument("--n_emb", type=int, default=config.get("n_emb", 5),
                        help="Embedding size. Default is 128.")
    parser.add_argument("--lr", type=float, default=config.get("lr", 0.01),
                        help="Learning rate. Increase for larger datasets.")
    parser.add_argument("--window_size", type=int, default=config.get("window_size", 3),
                        help="Context window size for pair generation. Default is 2.")
    parser.add_argument("--learn_fake_pos", type=bool, default=False,
                        help="Learn fake positive edges from negative generator. Default False")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=config.get("n_epochs", 1000),
                        help="Number of epochs. Default is 70.")
    parser.add_argument("--n_epochs_gen", type=int, default=config.get("n_epochs_gen", 5),
                        help="Generator inner loops per epoch. Default is 10.")
    parser.add_argument("--n_epochs_dis", type=int, default=config.get("n_epochs_dis", 5),
                        help="Discriminator inner loops per epoch. Default is 10.")
    parser.add_argument("--n_sample_gen", type=int, default=config.get("n_sample_gen", 2),
                        help="Number of samples per node. Default is 20.")

    # Batch sizes
    parser.add_argument("--batch_size_gen", type=int, default=config.get("batch_size_gen", 128),
                        help="Generator batch size. Default is 64.")
    parser.add_argument("--batch_size_dis", type=int, default=config.get("batch_size_dis", 128),
                        help="Discriminator batch size. Default is 64.")
    parser.add_argument("--n_node_subsets", type=int, default=1,
                        help="Number of node subsets for large graphs. Default is 1.")

    # Regularization and privacy
    parser.add_argument("--lambda_gen", type=float, default=1e-5,
                        help="Generator L2 regularization weight. Default is 1e-5.")
    parser.add_argument("--lambda_dis", type=float, default=1e-5,
                        help="Discriminator L2 regularization weight. Default is 1e-5.")
    parser.add_argument("--noise_stddev", type=float, default=config.get("noise_stddev", "2"),
                        help="Noise standard deviation for RDP.")
    parser.add_argument("--clip_value", type=float, default=config.get("clip_value", "1.0"),
                        help="Gradient clipping value for RDP.")
    parser.add_argument("--epsilon", type=float, default=3,
                        help="Privacy budget.")
    parser.add_argument('--delta', default=10 ** (-5),
                        help="Privacy delta parameter.")

    # Other parameters
    parser.add_argument("--update_ratio", type=int, default=1,
                        help="Tree update ratio. Default is 1.")
    parser.add_argument("--load_model", type=bool, default=False,
                        help="Load existing model for initialization. Default False")
    parser.add_argument("--skip_model_flag", type=bool, default=False,
                        help="Use skip model. Default False")
    parser.add_argument("--RDP", type=bool, default=True,
                        help="Enable RDP privacy. Default True")
    parser.add_argument("--partial_node_flag", type=bool, default=False,
                        help="Use partial nodes for large graphs. Default False")
    parser.add_argument("--save_steps", type=int, default=10,
                        help="Epoch interval for saving checkpoints. Default is 10.")

    args = parser.parse_args()

    # Set file paths
    args.train_filename = "../data/" + base_args.dataset + "/" + base_args.dataset + ".train"
    args.test_filename = "../data/" + base_args.dataset + "/" + base_args.dataset + ".test"

    # Set result file paths
    res_fn_path = "../results/" + base_args.dataset + "_dim" + str(args.n_emb) + "_lr" + str(args.lr)
    args.emb_filenames = [res_fn_path + "_gen_p", res_fn_path + "_dis_p",
                          res_fn_path + "_gen_n", res_fn_path + "_dis_n"]
    args.result_filename = res_fn_path + ".results"
    args.modes = ["gen_p", "dis_p", "gen_n", "dis_n"]
    args.model_log = "../log/"

    # Additional derived parameters
    args.gen_interval = args.n_epochs_gen
    args.dis_interval = args.n_epochs_dis
    args.lr_gen = args.lr
    args.lr_dis = args.lr
    args.task = 'node_clustering'
    print(args)
    print(f"*************Epsilon:{args.epsilon}*****************")
    return args


if __name__ == "__main__":
    config_all = parse_args()  # Parse arguments and load config
    asgl = ASGL(config_all)  # Initialize ASiNE model
    asgl.train()  # Start training
