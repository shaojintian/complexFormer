import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans # 用于初始化 M 和 C (可选)
import numpy as np

__all__ = ['VQLogits']

class VQLogits(nn.Module):
    def __init__(self, d_model, vocab_size, K, M_init_method="random", C_init_from_embeddings=None):
        """
        VQ-Logits Layer.

        Args:
            d_model (int): Dimension of the hidden state from the LLM.
            vocab_size (int): Full vocabulary size (V).
            K (int): Number of codebook vectors (K << V).
            M_init_method (str): Method to initialize mapping M.
                                 "random": Randomly assign vocab tokens to K codes.
                                 "kmeans_on_C_init": If C_init_from_embeddings is provided,
                                                     use k-means cluster assignments.
                                 "contiguous_blocks": Divide vocab into K contiguous blocks.
            C_init_from_embeddings (torch.Tensor, optional): Pre-trained full output embeddings
                                                            (shape: V x d_model) to initialize C and M.
                                                            If None, C is randomly initialized.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.K = K

        # 1. Initialize Codebook (C)
        # C is a learnable matrix of K code vectors, each of dimension d_model
        # Shape: (K, d_model)
        if C_init_from_embeddings is not None and M_init_method == "kmeans_on_C_init":
            print(f"Initializing Codebook C and Mapping M using k-means on provided embeddings...")
            if C_init_from_embeddings.shape[0] != vocab_size or C_init_from_embeddings.shape[1] != d_model:
                raise ValueError(f"C_init_from_embeddings shape mismatch. Expected ({vocab_size}, {d_model}), "
                                 f"got {C_init_from_embeddings.shape}")

            # Perform k-means clustering
            # Ensure embeddings are on CPU and numpy for sklearn
            embeddings_np = C_init_from_embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(embeddings_np)

            # Centroids become the initial codebook C
            initial_C = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.codebook_C = nn.Parameter(initial_C) # Shape: (K, d_model)

            # Cluster assignments become the mapping M
            initial_M = torch.tensor(kmeans.labels_, dtype=torch.long) # Shape: (V)
            self.register_buffer('mapping_M', initial_M) # Not learnable, fixed after init
            print("Initialization of C and M from k-means complete.")

        else:
            print(f"Initializing Codebook C randomly.")
            self.codebook_C = nn.Parameter(torch.randn(K, d_model))
            # Initialize M separately if C is random or k-means not used for M
            self._initialize_M(M_init_method, C_init_from_embeddings)

        if self.mapping_M is None: # Fallback if M wasn't set by k-means
             self._initialize_M(M_init_method, C_init_from_embeddings)


    def _initialize_M(self, M_init_method, C_init_from_embeddings):
        """ Helper to initialize mapping M if not done via k-means on C_init_from_embeddings """
        print(f"Initializing Mapping M using method: {M_init_method}")
        initial_M = torch.zeros(self.vocab_size, dtype=torch.long)
        if M_init_method == "random":
            initial_M = torch.randint(0, self.K, (self.vocab_size,), dtype=torch.long)
        elif M_init_method == "contiguous_blocks":
            # Divide vocab into K contiguous blocks
            block_size = (self.vocab_size + self.K - 1) // self.K # Ceiling division
            for i in range(self.vocab_size):
                initial_M[i] = i // block_size
            initial_M = torch.clamp(initial_M, 0, self.K - 1) # Ensure last block maps correctly
        elif M_init_method == "kmeans_on_input_embeddings" and C_init_from_embeddings is not None:
            # This assumes C_init_from_embeddings are actually *input* embeddings if C itself is random
            # Or, more appropriately, if you have separate input embeddings to cluster for M
            print(f"Initializing Mapping M using k-means on provided embeddings (for M)...")
            if C_init_from_embeddings.shape[0] != self.vocab_size or C_init_from_embeddings.shape[1] != self.d_model:
                raise ValueError(f"Embeddings for M init shape mismatch. Expected ({self.vocab_size}, {self.d_model}), "
                                 f"got {C_init_from_embeddings.shape}")
            embeddings_np = C_init_from_embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.K, random_state=0, n_init='auto').fit(embeddings_np)
            initial_M = torch.tensor(kmeans.labels_, dtype=torch.long)
        elif M_init_method == "kmeans_on_C_init" and C_init_from_embeddings is None:
            print("Warning: M_init_method='kmeans_on_C_init' but C_init_from_embeddings is None. Defaulting M to random.")
            initial_M = torch.randint(0, self.K, (self.vocab_size,), dtype=torch.long)
        else:
            print(f"Warning: Unknown M_init_method '{M_init_method}' or missing embeddings. Defaulting M to random.")
            initial_M = torch.randint(0, self.K, (self.vocab_size,), dtype=torch.long)

        self.register_buffer('mapping_M', initial_M) # Shape: (V), not learnable
        print("Initialization of M complete.")


    def forward(self, hidden_states):
        """
        Forward pass.

        Args:
            hidden_states (torch.Tensor): Final hidden states from the LLM.
                                          Shape: (B, S, d_model)
                                          B: batch_size, S: sequence_length

        Returns:
            torch.Tensor: Logits over the full vocabulary.
                          Shape: (B, S, V)
        """
        B, S, D_model = hidden_states.shape
        if D_model != self.d_model:
            raise ValueError(f"hidden_states d_model ({D_model}) does not match "
                             f"layer's d_model ({self.d_model})")

        # Reshape hidden_states for matrix multiplication if needed,
        # but direct matmul works fine if last dim matches
        # hidden_states_flat = hidden_states.view(-1, self.d_model) # Shape: (B*S, d_model)

        # 1. Compute Codebook Logits (L_c)
        # L_c = h * C^T
        # hidden_states: (B, S, d_model)
        # self.codebook_C.t(): (d_model, K)
        # codebook_logits_Lc: (B, S, K)
        codebook_logits_Lc = torch.matmul(hidden_states, self.codebook_C.t())

        # 2. Scatter Logits (L_v)
        # Expand codebook logits to full vocabulary size using the mapping M.
        # For each vocabulary token vi, its logit Lv[..., i] is the logit of its
        # assigned codebook vector c_M(i).
        # Lv[b, s, i] = Lc[b, s, M(i)]

        # Efficient scatter using gather (or advanced indexing)
        # codebook_logits_Lc needs to be (B*S, K) for gather along dim 1
        # mapping_M needs to be (B*S, V) where each row is self.mapping_M
        # A more direct way:
        #   - Lc_expanded_for_vocab: (B, S, K) -> (B, S, V) by selecting from K using M

        # Ensure mapping_M is on the same device as codebook_logits_Lc
        current_device = codebook_logits_Lc.device
        if self.mapping_M.device != current_device:
            self.mapping_M = self.mapping_M.to(current_device)

        # Flatten Lc for easier gathering
        # codebook_logits_Lc_flat = codebook_logits_Lc.view(-1, self.K) # (B*S, K)
        # mapping_M_expanded = self.mapping_M.unsqueeze(0).expand(B * S, -1) # (B*S, V)
        # full_vocab_logits_Lv_flat = torch.gather(codebook_logits_Lc_flat, 1, mapping_M_expanded)
        # This is incorrect, gather expects indices to select *from* the K dimension.

        # Correct scatter:
        # For each vocab item i, we want to pick the logit Lc[M[i]]
        # Lc is (B, S, K)
        # M is (V)
        # Lv will be (B, S, V)
        # Lv[:, :, i] = Lc[:, :, M[i]] for all i in V

        # Using advanced indexing:
        #   - Select the appropriate K-dimensional logit vector for each vocabulary item
        #   - self.mapping_M provides the indices into the K dimension of codebook_logits_Lc
        #   - Example: if M[vocab_idx] = code_idx, then Lv[..., vocab_idx] = Lc[..., code_idx]
        #   - We can use self.mapping_M to index the last dimension of codebook_logits_Lc
        #     `codebook_logits_Lc[:, :, self.mapping_M]`
        #     This means for each (b,s), we take codebook_logits_Lc[b,s,:] which is a K-dim vector,
        #     and then we use self.mapping_M (which has V elements, values from 0 to K-1)
        #     to select V elements from this K-dim vector.
        #     The result will be (B, S, V)

        full_vocab_logits_Lv = codebook_logits_Lc[:, :, self.mapping_M]

        # The above line is a concise way to do it:
        # For each (b, s) pair:
        #   lc_slice = codebook_logits_Lc[b, s, :]  (shape K)
        #   lv_slice = torch.zeros(self.vocab_size, device=current_device)
        #   for vocab_idx in range(self.vocab_size):
        #       code_idx = self.mapping_M[vocab_idx]
        #       lv_slice[vocab_idx] = lc_slice[code_idx]
        #   full_vocab_logits_Lv[b, s, :] = lv_slice
        # The one-liner `codebook_logits_Lc[:, :, self.mapping_M]` does this efficiently.

        # 3. Full Softmax (typically done outside this module, in the loss function or main model)
        # P = softmax(Lv)
        # For this module, we just return the logits Lv

        return full_vocab_logits_Lv

# --- Example Usage ---
if __name__ == '__main__':
    B = 2  # Batch size
    S = 10 # Sequence length
    D_MODEL = 64 # Dimension of hidden state
    VOCAB_SIZE = 1000 # Full vocabulary size
    K_CODES = 50    # Number of codebook vectors
    
    # --- Option 1: Random Initialization ---
    print("--- Testing with Random Initialization ---")
    vq_logits_random = VQLogits(D_MODEL, VOCAB_SIZE, K_CODES, M_init_method="random")
    hidden_states_test = torch.randn(B, S, D_MODEL)
    output_logits_random = vq_logits_random(hidden_states_test)
    print(f"Input hidden_states shape: {hidden_states_test.shape}")
    print(f"Output logits shape (random init): {output_logits_random.shape}") # Expected: (B, S, VOCAB_SIZE)
    assert output_logits_random.shape == (B, S, VOCAB_SIZE)

    # Check if scatter worked as expected (values in Lv come from Lc)
    # For a specific batch and sequence position:
    b, s_idx = 0, 0
    lc_sample = torch.matmul(hidden_states_test[b, s_idx, :], vq_logits_random.codebook_C.t()) # (K_CODES)
    lv_sample_output = output_logits_random[b, s_idx, :] # (VOCAB_SIZE)
    mapping_M_sample = vq_logits_random.mapping_M

    is_correct_scatter = True
    for vocab_i in range(VOCAB_SIZE):
        code_idx = mapping_M_sample[vocab_i]
        if not torch.isclose(lv_sample_output[vocab_i], lc_sample[code_idx]):
            is_correct_scatter = False
            print(f"Mismatch at vocab_i={vocab_i}: Lv={lv_sample_output[vocab_i]}, Lc[M[i]]={lc_sample[code_idx]}")
            break
    print(f"Scatter logic seems correct (random init): {is_correct_scatter}")
    print("-" * 30)

    # --- Option 2: K-means Initialization (if pre-trained embeddings are available) ---
    print("\n--- Testing with K-means Initialization ---")
    # Simulate pre-trained output embeddings (rows are token embeddings)
    # These would typically come from a fully trained model with a standard output layer
    # For VQ-Logits, E_out is W_out.T, so shape V x d_model
    pretrained_E_out_simulated = torch.randn(VOCAB_SIZE, D_MODEL) * 5 # Make them a bit spread out

    vq_logits_kmeans = VQLogits(D_MODEL,
                                VOCAB_SIZE,
                                K_CODES,
                                M_init_method="kmeans_on_C_init", # This uses the embeddings for C and M
                                C_init_from_embeddings=pretrained_E_out_simulated)

    hidden_states_test_2 = torch.randn(B, S, D_MODEL)
    output_logits_kmeans = vq_logits_kmeans(hidden_states_test_2)
    print(f"Input hidden_states shape: {hidden_states_test_2.shape}")
    print(f"Output logits shape (k-means init): {output_logits_kmeans.shape}") # Expected: (B, S, VOCAB_SIZE)
    assert output_logits_kmeans.shape == (B, S, VOCAB_SIZE)

    # Verify mapping_M and codebook_C values if k-means was run
    if hasattr(vq_logits_kmeans, 'mapping_M') and vq_logits_kmeans.mapping_M is not None:
        print(f"Mapping M (first 10): {vq_logits_kmeans.mapping_M[:10]}")
        print(f"Codebook C shape: {vq_logits_kmeans.codebook_C.shape}") # Expected (K_CODES, D_MODEL)

    b, s_idx = 0, 0
    lc_sample_kmeans = torch.matmul(hidden_states_test_2[b, s_idx, :], vq_logits_kmeans.codebook_C.t()) # (K_CODES)
    lv_sample_output_kmeans = output_logits_kmeans[b, s_idx, :] # (VOCAB_SIZE)
    mapping_M_sample_kmeans = vq_logits_kmeans.mapping_M

    is_correct_scatter_kmeans = True
    for vocab_i in range(VOCAB_SIZE):
        code_idx = mapping_M_sample_kmeans[vocab_i]
        if not torch.isclose(lv_sample_output_kmeans[vocab_i], lc_sample_kmeans[code_idx]):
            is_correct_scatter_kmeans = False
            print(f"Mismatch at vocab_i={vocab_i}: Lv={lv_sample_output_kmeans[vocab_i]}, Lc[M[i]]={lc_sample_kmeans[code_idx]}")
            break
    print(f"Scatter logic seems correct (k-means init): {is_correct_scatter_kmeans}")
    print("-" * 30)


        # --- Standard Output Layer (for comparison of parameter count) ---
    class StandardOutputLayer(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            # W_out is effectively linear.weight.T in terms of parameters,
            # or just linear.weight if you consider W_out to be (V, d_model)
            # nn.Linear(in_features, out_features) has weight of shape (out_features, in_features)
            # So, to map from d_model to vocab_size, weight is (vocab_size, d_model)
            # If we consider W_out to be (d_model, vocab_size) as in the paper's math L = h * W_out,
            # then a linear layer mapping d_model to vocab_size has its weight matrix as W_out^T.
            # The number of parameters is the same: d_model * vocab_size.
            self.linear = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, hidden_states):
            return self.linear(hidden_states)

    standard_layer = StandardOutputLayer(D_MODEL, VOCAB_SIZE)
    num_params_standard = sum(p.numel() for p in standard_layer.parameters() if p.requires_grad)
    # For VQLogits, only codebook_C is a learnable parameter
    num_params_vq = sum(p.numel() for p in vq_logits_kmeans.parameters() if p.requires_grad)
    # M is a buffer, not a parameter with requires_grad=True

    print(f"\nParameter Count Comparison:")
    print(f"Standard Output Layer (W_out parameters: d_model * vocab_size):")
    print(f"  - d_model: {D_MODEL}, vocab_size: {VOCAB_SIZE}")
    print(f"  - Total learnable parameters: {num_params_standard} (Expected: {D_MODEL*VOCAB_SIZE})")

    print(f"\nVQ-Logits Layer (Codebook C parameters: K * d_model):")
    print(f"  - d_model: {D_MODEL}, K_codes: {K_CODES}")
    print(f"  - Total learnable parameters: {num_params_vq} (Expected: {K_CODES*D_MODEL})")
    if num_params_standard > 0 :
        reduction_factor = num_params_standard / num_params_vq if num_params_vq > 0 else float('inf')
        reduction_percentage = ((num_params_standard - num_params_vq) / num_params_standard) * 100
        print(f"  - Parameter reduction factor: ~{reduction_factor:.2f}x")
        print(f"  - Parameter reduction percentage: ~{reduction_percentage:.2f}%")

    print(f"\nNon-learnable Mapping M (VQ-Logits):")
    print(f"  - Elements: {vq_logits_kmeans.mapping_M.numel()} (vocab_size)")
    print(f"  - Memory for M (approx. if int32): {vq_logits_kmeans.mapping_M.numel() * 4} bytes (assuming int32/long for indices)")