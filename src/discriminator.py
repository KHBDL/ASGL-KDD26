import tensorflow as tf

class Discriminator(object):
    def __init__(self, n_node, node_emb_init, positive, config):
        # Initialize discriminator with number of nodes, initial embeddings, positive flag, and config
        self.n_node = n_node  # Number of nodes in graph
        self.node_emb_init = node_emb_init  # Initial node embeddings

        with tf.variable_scope('discriminator'):
            # Create trainable embedding matrix for discriminator
            self.embedding_matrix = tf.get_variable(name="embedding_discriminator",
                                                  shape=self.node_emb_init.shape,
                                                  trainable=True)
            if config.skip_model_flag:
                # Additional embedding matrix for skip-gram model if enabled
                self.embedding_matrix_context = tf.get_variable(
                    name="embedding_discriminator_context",
                    shape=self.node_emb_init.shape,
                    trainable=True)
            # Initialize bias vector with zeros
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        # Define input placeholders
        self.node_id = tf.placeholder(tf.int32, shape=[None])  # Source node IDs
        self.node_neighbor_id = tf.placeholder(tf.int32, shape=[None])  # Neighbor node IDs
        self.label = tf.placeholder(tf.float32, shape=[None])  # Edge labels (1 for real, 0 for fake)

        # Get embeddings for source nodes
        self.node_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.node_id)
        # Get embeddings for neighbor nodes (using context matrix if skip_model_flag is True)
        if config.skip_model_flag:
            self.node_neighbor_embedding = tf.nn.embedding_lookup(
                self.embedding_matrix_context,
                self.node_neighbor_id)
        else:
            self.node_neighbor_embedding = tf.nn.embedding_lookup(
                self.embedding_matrix,
                self.node_neighbor_id)

        # Get bias values for neighbor nodes
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)
        # Calculate similarity score between node pairs (dot product + bias)
        self.score = tf.reduce_sum(
            tf.multiply(self.node_embedding, self.node_neighbor_embedding),
            axis=1) + self.bias

        if positive == True:
            # Loss for positive discriminator: standard sigmoid cross entropy
            self.loss = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.label,
                    logits=self.score)) + \
                config.lambda_dis * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) +
                    tf.nn.l2_loss(self.node_embedding) +
                    tf.nn.l2_loss(self.bias))
            if config.skip_model_flag:
                # Additional skip-gram loss component if enabled
                self.loss_sgm = -tf.reduce_mean(tf.log_sigmoid(self.score))
                self.loss = self.loss_sgm + 0.5 * self.loss
        else:
            # Loss for negative discriminator: negative sigmoid cross entropy
            self.loss = tf.reduce_sum(
                -tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.label,
                    logits=self.score)) + \
                config.lambda_dis * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) +
                    tf.nn.l2_loss(self.node_embedding) +
                    tf.nn.l2_loss(self.bias))
            if config.skip_model_flag:
                # Additional skip-gram loss component if enabled
                self.loss_sgm = -tf.reduce_mean(tf.log_sigmoid(-self.score))
                self.loss = self.loss_sgm + 0.5 * self.loss

        # Operations for calculating similarity between target nodes and all others
        self.target_node = tf.placeholder(tf.int32, shape=[None])  # Target node IDs
        self.target_embedding = tf.nn.embedding_lookup(
            self.embedding_matrix,
            self.target_node)
        if config.skip_model_flag:
            self.target_score = tf.matmul(
                self.target_embedding,
                self.embedding_matrix_context,
                transpose_b=True) + self.bias_vector
        else:
            self.target_score = tf.matmul(
                self.target_embedding,
                self.embedding_matrix,
                transpose_b=True) + self.bias_vector

        if config.RDP:
            # Differential privacy implementation
            # 1. Define optimizer
            self.optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
            self.params = [v for v in tf.trainable_variables()
                         if 'discriminator' in v.name]

            # 2. Compute gradients and add noise for differential privacy
            self.var_list = self.params
            self.noise_stddev = config.noise_stddev
            self.grads_and_vars = self.optimizer.compute_gradients(
                self.loss,
                self.var_list)

            # Process each gradient
            for i, (g, v) in enumerate(self.grads_and_vars):
                if g is not None:
                    # Clip gradients
                    g = tf.clip_by_norm(g, config.clip_value)
                    # Calculate noise scaling factor
                    K = (config.n_sample_gen**config.window_size - 1) / \
                        (config.n_sample_gen - 1)
                    # Add Gaussian noise
                    noise = tf.random_normal(
                        shape=tf.shape(g),
                        stddev=self.noise_stddev * config.clip_value * K)
                    noisy_g = g + noise
                    self.grads_and_vars[i] = (noisy_g, v)
                else:
                    self.grads_and_vars[i] = (None, v)

            # 4. Apply noisy gradients
            self.d_updates = self.optimizer.apply_gradients(self.grads_and_vars)
        else:
            # Standard optimization without differential privacy
            optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
            self.d_updates = optimizer.minimize(self.loss)

        # Additional operations
        self.score = tf.clip_by_value(
            self.score,
            clip_value_min=-10,
            clip_value_max=10)
        # Calculate reward using softplus function
        self.reward = tf.log(1 + tf.exp(self.score))