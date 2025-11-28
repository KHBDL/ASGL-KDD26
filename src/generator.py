import tensorflow as tf


class Generator(object):
    def __init__(self, n_node, node_emb_init, positive, config):
        # Initialize generator with number of nodes, initial embeddings, positive flag, and config
        self.n_node = n_node  # Total number of nodes in graph
        self.node_emb_init = node_emb_init  # Initial node embeddings

        with tf.variable_scope('generator'):
            # Create trainable embedding matrix in generator scope
            self.embedding_matrix = tf.get_variable(name="embedding_generator",
                                                    shape=self.node_emb_init.shape,
                                                    trainable=True)
            # Initialize bias vector with zeros
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        # Define input placeholders
        self.node_id = tf.placeholder(tf.int32, shape=[None])  # Source node IDs
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])  # Neighbor node IDs
        self.reward = tf.placeholder(tf.float32, shape=[None])  # Reward from discriminator

        # Get embeddings for source nodes
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)

        # Target node related placeholders and operations
        self.target_node = tf.placeholder(tf.int32, shape=[None])  # Target node IDs
        # Get embeddings for target nodes
        self.target_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.target_node)
        # Calculate similarity scores between target nodes and all other nodes
        self.target_score = tf.matmul(self.target_embedding,
                                      self.embedding_matrix,
                                      transpose_b=True) + self.bias_vector

        # Get embeddings for neighbor nodes
        self.node_neighbor_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                              self.node_neighbor_id)
        # Get bias values for neighbor nodes
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)

        # Calculate score for node pairs (dot product + bias)
        self.score = tf.reduce_sum(self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        # Convert scores to probabilities using sigmoid (clipped for numerical stability)
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        # Placeholder for precomputed pair relevance scores
        self.pairs_relevances = tf.placeholder(tf.float32, shape=[None])
        # Convert relevance scores to probabilities
        self.pairs_score = tf.clip_by_value(tf.nn.sigmoid(self.pairs_relevances), 1e-5, 1)

        if positive == True:
            # Loss for positive samples: minimize negative log likelihood weighted by reward
            # Equivalent to maximizing E[log(prob) * reward] to encourage positive sample generation
            self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + \
                        config.lambda_gen * (tf.nn.l2_loss(self.node_neighbor_embedding) +
                                             tf.nn.l2_loss(self.node_embedding))
        else:
            # Loss for negative samples: maximize log likelihood weighted by reward
            # Equivalent to minimizing E[log(prob) * reward] to discourage negative sample generation
            self.loss = tf.reduce_mean(tf.log(self.prob) * self.reward) + \
                        config.lambda_gen * (tf.nn.l2_loss(self.node_neighbor_embedding) +
                                             tf.nn.l2_loss(self.node_embedding))

        # Create optimizer and training operation
        optimizer = tf.train.GradientDescentOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)