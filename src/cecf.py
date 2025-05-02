import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralClustering
from utils import get_metric, get_user_seqs
import time

# No Clustering
# python cecf.py --data_name=Beauty --do_eval --no_cluster

# SVD KMeans
# python cecf.py --data_name=Beauty --do_eval --cluster_method=svd_kmeans --n_clusters_list 149 500 1000 5000 --top_k=10

# Spectral Clustering
# python cecf.py --data_name=Beauty --do_eval --cluster_method=spectral --n_clusters_list 149 500 1000 --top_k=10

def build_recency_matrix(user_seq, time_seq, item_size):
    U = len(user_seq)
    R = np.zeros((U, item_size), dtype=np.float32)
    for u, (full_items, full_times) in enumerate(zip(user_seq, time_seq)):
        items = full_items[:-2]
        times = full_times[:-2]
        if not items:
            continue
        t_min, t_max = min(times), max(times)
        span = t_max - t_min if t_max != t_min else 1
        for i, t in zip(items, times):
             if i < item_size:
                R[u, i] = (t - t_min) / span
    return R

def stochastic_convert(M):
    S = M.copy()
    rowsum = S.sum(axis=1, keepdims=True)
    rowsum[rowsum==0] = 1
    return S / rowsum

def cluster_via_svd_kmeans(M, n_clusters, seed, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z   = svd.fit_transform(M)
    km  = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = km.fit_predict(Z)
    U, V = len(labels), M.shape[1]
    eps = 0.01
    uniform = np.ones(V, dtype=np.float32)/V
    profiles = np.zeros((n_clusters, V), dtype=np.float32)
    for c in range(n_clusters):
        members = np.where(labels==c)[0]
        if len(members)>0:
            raw = M[members].mean(axis=0)
            raw = raw / (raw.sum()+1e-9)
            profiles[c] = (1-eps)*raw + eps*uniform
    return profiles, labels

def cluster_via_spectral(M, n_clusters, seed, affinity='rbf', gamma=0.005):
    spec = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        gamma=gamma,
        assign_labels='kmeans',
        random_state=seed,
        n_jobs=-1
    )
    labels = spec.fit_predict(M)

    U, V = M.shape
    eps = 0.01
    uniform = np.ones(V, dtype=np.float32) / V
    profiles = np.zeros((n_clusters, V), dtype=np.float32)

    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if len(members)>0:
            raw = M[members].mean(axis=0)
            raw = raw / (raw.sum()+1e-9)
            profiles[c] = (1-eps)*raw + eps*uniform
    return profiles, labels

def build_recommender(user_vecs, cluster_profiles, train_matrix, K=10, topn=20):
    sim    = cosine_similarity(user_vecs, cluster_profiles)
    scores = sim @ cluster_profiles

    scores[ train_matrix.nonzero() ] = -np.inf

    U, V = scores.shape
    ind   = np.argpartition(scores, -topn, axis=1)[:, -topn:]
    vals  = scores[np.arange(U)[:,None], ind]
    order = np.argsort(vals, axis=1)[:, ::-1]
    topn_list = ind[np.arange(U)[:,None], order]

    return topn_list[:, :K]

def get_rank(pred, truth):
    if truth is None:
        return None
    for i, item in enumerate(pred):
        if item == truth:
            return i
    return len(pred)

def get_item(user_seq, time_seq, index, data_type, not_aug_users=None):
    """
    Mimics RecWithContrastiveLearningDataset.__getitem__ for retrieval of
    the held-out answer (valid or test) without spinning up a full Dataset.
    Returns: (input_ids, target_pos, answer_list)
    """
    items = user_seq[index]
    times = time_seq[index]

    if data_type == "train":
        input_ids  = items[:-3]
        target_pos = items[1:-2]
        answer     = [0]
    elif data_type == "valid":
        input_ids  = items[:-2]
        target_pos = items[1:-1]
        answer     = [items[-2]]
    elif data_type == "test":
        input_ids  = items[:-1]
        target_pos = items[1:]
        answer     = [items[-1]]
    else:
        raise ValueError(f"Unknown data_type={data_type}")

    return input_ids, target_pos, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    type=str,   default='../data')
    parser.add_argument('--data_name',   type=str,   default='Beauty')
    parser.add_argument('--seed',        type=int,   default=2022)
    # Change n_clusters to have a default, but it might be overridden by the list
    parser.add_argument('--n_clusters',  type=int,   default=100, help="Default number of clusters if not using list.")
    # Add argument for list of cluster numbers
    parser.add_argument('--n_clusters_list', type=int, nargs='+', default=None,
                        help="List of n_clusters values for grid search.")
    parser.add_argument('--top_k',       type=int,   default=10)
    parser.add_argument('--do_eval',     action='store_true')
    parser.add_argument('--no_cluster',  action='store_true',
                        help="skip clustering; use each user vector as its own profile")
    parser.add_argument('--cluster_method',
                        choices=['svd_kmeans','spectral'],
                        default='spectral', # Keep default, but grid search only works for svd_kmeans
                        help="which clustering algorithm to use")
    parser.add_argument("--var_rank_not_aug_ratio", type=float, default=0.15,
                        help="Percentage of training samples that will not be augmented")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- Data Loading and Preprocessing ---
    print("Loading data...")
    args.data_file = os.path.join(args.data_dir, args.data_name,
                             args.data_name + '_item.txt')
    args.time_file = os.path.join(args.data_dir, args.data_name,
                             args.data_name + '_time.txt')

    user_seq, time_seq, max_item, valid_rating_matrix, test_rating_matrix, not_aug_users = get_user_seqs(args)

    U = len(user_seq)
    V = max_item + 1

    print("Building recency matrix and user vectors...")
    R = build_recency_matrix(user_seq, time_seq, V)
    S = stochastic_convert(R)
    Uvecs = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)

    # --- Prepare Ground Truth ---
    valid_truth = []
    for u in range(U):
        _, _, ans = get_item(user_seq, time_seq, u, data_type='valid')
        valid_truth.append(ans[0] if ans else None)

    test_truth = []
    for u in range(U):
        _, _, ans = get_item(user_seq, time_seq, u, data_type='test')
        test_truth.append(ans[0] if ans else None)

    valid_idx = [u for u, t in enumerate(valid_truth) if t is not None]
    test_idx  = [u for u, t in enumerate(test_truth ) if t is not None]

    # --- Evaluation Logic ---
    if args.do_eval:
        results = {} # Store results for grid search

        # Determine which cluster numbers to test
        cluster_numbers_to_test = []
        if args.n_clusters_list is not None:
            if args.no_cluster:
                print("Warning: --n_clusters_list provided but --no_cluster is set. Ignoring cluster list.")
            else:
                cluster_numbers_to_test = sorted(list(set(args.n_clusters_list))) # Use unique sorted list
        elif not args.no_cluster:
             cluster_numbers_to_test = [args.n_clusters] # Use the single value if no list provided

        # --- Run Baseline (No Clustering) ---
        if args.no_cluster or args.n_clusters_list is not None: # Also run baseline if doing grid search for comparison
            print("\n--- Running Evaluation: No Clustering ---")
            start_time = time.time()
            profiles_nc = Uvecs
            recs_val_nc  = build_recommender(
                            Uvecs, profiles_nc, train_matrix=valid_rating_matrix,
                            K=args.top_k, topn=max(args.top_k,20)
                        )
            recs_test_nc = build_recommender(
                            Uvecs, profiles_nc, train_matrix=test_rating_matrix,
                            K=args.top_k, topn=max(args.top_k,20)
                        )
            val_ranks_nc  = [get_rank(recs_val_nc[u],  valid_truth[u]) for u in valid_idx]
            test_ranks_nc = [get_rank(recs_test_nc[u], test_truth[u])  for u in test_idx]
            val_hit_nc,  val_ndcg_nc,  val_mrr_nc  = get_metric(val_ranks_nc,  args.top_k)
            test_hit_nc, test_ndcg_nc, test_mrr_nc = get_metric(test_ranks_nc, args.top_k)
            duration = time.time() - start_time
            print(f"(No Cluster) Val Hit@{args.top_k}={val_hit_nc:.4f} | NDCG@{args.top_k}={val_ndcg_nc:.4f} | MRR={val_mrr_nc:.4f}")
            print(f"(No Cluster) Test Hit@{args.top_k}={test_hit_nc:.4f} | NDCG@{args.top_k}={test_ndcg_nc:.4f} | MRR={test_mrr_nc:.4f}")
            print(f"(No Cluster) Time: {duration:.2f}s")
            results['no_cluster'] = {'val': (val_hit_nc, val_ndcg_nc, val_mrr_nc), 'test': (test_hit_nc, test_ndcg_nc, test_mrr_nc), 'time': duration}
            if args.no_cluster and args.n_clusters_list is None: # Exit if only --no_cluster was requested
                 return


        # --- Run Clustering Methods ---
        if cluster_numbers_to_test:
            print(f"\n--- Running Evaluation: {args.cluster_method} ---")
            k_svd = min(100, S.shape[1]-1) # Precompute SVD components if needed

            for n_clust in cluster_numbers_to_test:
                print(f"\n--- Testing n_clusters = {n_clust} ---")
                start_time = time.time()
                profiles, labels = None, None

                if args.cluster_method == 'svd_kmeans':
                    if n_clust >= U:
                         print(f"Skipping n_clusters={n_clust} as it's >= number of users ({U})")
                         continue
                    if n_clust <= 1:
                         print(f"Skipping n_clusters={n_clust} as it must be > 1")
                         continue
                    print(f"Clustering (svd_kmeans, k={k_svd})...")
                    profiles, labels = cluster_via_svd_kmeans(S, n_clust, args.seed, n_components=k_svd)
                elif args.cluster_method == 'spectral':
                     # This part remains unchanged, will only run if spectral is selected AND n_clusters_list is None
                    if n_clust >= U:
                         print(f"Skipping n_clusters={n_clust} as it's >= number of users ({U})")
                         continue
                    if n_clust <= 1:
                         print(f"Skipping n_clusters={n_clust} as it must be > 1")
                         continue
                    print("Clustering (spectral)...")
                    profiles, labels = cluster_via_spectral(
                        S, n_clust, args.seed, affinity='rbf', gamma=0.005
                    )
                else:
                    # Should not happen due to argument choices, but good practice
                    print(f"Error: Unknown cluster method '{args.cluster_method}'")
                    continue # Skip to next cluster number

                print("Building recommendations...")
                recs_val  = build_recommender(
                                Uvecs, profiles, train_matrix=valid_rating_matrix,
                                K=args.top_k, topn=max(args.top_k,20)
                            )
                recs_test = build_recommender(
                                Uvecs, profiles, train_matrix=test_rating_matrix,
                                K=args.top_k, topn=max(args.top_k,20)
                            )

                print("Calculating metrics...")
                val_ranks  = [get_rank(recs_val[u],  valid_truth[u]) for u in valid_idx]
                test_ranks = [get_rank(recs_test[u], test_truth[u])  for u in test_idx]

                val_hit,  val_ndcg,  val_mrr  = get_metric(val_ranks,  args.top_k)
                test_hit, test_ndcg, test_mrr = get_metric(test_ranks, args.top_k)
                duration = time.time() - start_time

                print(f"(n={n_clust}) Val Hit@{args.top_k}={val_hit:.4f} | NDCG@{args.top_k}={val_ndcg:.4f} | MRR={val_mrr:.4f}")
                print(f"(n={n_clust}) Test Hit@{args.top_k}={test_hit:.4f} | NDCG@{args.top_k}={test_ndcg:.4f} | MRR={test_mrr:.4f}")
                print(f"(n={n_clust}) Time: {duration:.2f}s")
                results[n_clust] = {'val': (val_hit, val_ndcg, val_mrr), 'test': (test_hit, test_ndcg, test_mrr), 'time': duration}

        # --- Print Summary of Grid Search ---
        if args.n_clusters_list is not None:
            print(f"\n--- Grid Search Summary {args.cluster_method} ---")
            best_test_hit = -1
            best_n_clust = None
            if 'no_cluster' in results:
                 nc_hit, _, _ = results['no_cluster']['test']
                 print(f"Baseline (No Cluster): Test Hit@{args.top_k} = {nc_hit:.4f}")
                 best_test_hit = nc_hit
                 best_n_clust = 'no_cluster'


            for n_clust, res in results.items():
                if isinstance(n_clust, int): # Only compare cluster results
                    hit, _, _ = res['test']
                    print(f"n_clusters = {n_clust:<5}: Test Hit@{args.top_k} = {hit:.4f}")
                    if hit > best_test_hit:
                        best_test_hit = hit
                        best_n_clust = n_clust

            print(f"\nBest result: n_clusters = {best_n_clust} with Test Hit@{args.top_k} = {best_test_hit:.4f}")

        return # End evaluation section

    # --- Default message if not evaluating ---
    print("Clustering/Vector preparation complete.")
    print("Pass `--do_eval` to evaluate performance.")
    if args.n_clusters_list:
         print("Pass `--do_eval --cluster_method=svd_kmeans` to run grid search.")


if __name__ == "__main__":
    main()