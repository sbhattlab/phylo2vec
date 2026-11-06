//! Tree-level objective functions for Phylo2Vec vectors: balanced minimum evolution (BME).
//!
//! BME scores an unrooted topology against a distance matrix `D` with Pauplin's tree length
//!
//! ```text
//! L(T) = sum_{i < j} 2^(1 - tau_ij) * D_ij
//! ```
//!
//! where `tau_ij` is the number of edges between leaves `i` and `j`.
//! The FastME/Pauplin BME branch lengths are the edge lengths whose sum is `L(T)`.
//!
//! Both quantities are computed from the weighted distance across each edge,
//! `S(e) = sum over the pairs that e separates of 2^(1 - tau_ij) * D_ij`.
//! A pair contributes `+2^(1 - tau_ij) D_ij` to every edge on its path,
//! and `-2^(1 - tau_ij) D_ij` to every edge hanging off an internal node of its path.
//! This gives `l(u, w) = 3 S(e) - T(u) - T(w)` for an internal edge
//! and `l(u, w) = 2 S(e) - T(w)` for a pendant edge to leaf `u`,
//! where `T(u)` is the weight of the pairs whose path passes through internal node `u`
//! (half the sum of `S` over its three edges).
//! Visiting each leaf pair once, at its lowest common ancestor, yields all `S(e)`
//! in O(n^2) time and O(n) extra memory.
use ndarray::ArrayView2;

use crate::types::Ancestry;
use crate::vector::convert::to_ancestry;

/// Panics unless `d` is an `n x n` matrix for the `n = v.len() + 1` leaves encoded by `v`.
fn check_distances(v: &[usize], d: &ArrayView2<f64>) {
    let n = v.len() + 1;
    assert!(
        d.shape() == [n, n],
        "Distance matrix must be {n} x {n} for a tree with {n} leaves, got {:?}",
        d.shape()
    );
}

/// For every edge, the sum of `2^(1 - tau_ij) * D_ij` over the leaf pairs it separates,
/// and the BME length (the same sum over all pairs).
///
/// Edges are indexed by the node below them.
/// Merging subtrees in ancestry order (children before parents) visits each pair once,
/// at its lowest common ancestor `a`.
/// Instead of walking the pair's path, it leaves marks at the path's ends
/// (`+w` at both leaves, `-2w` at `a`).
/// A final pass in the same order sums the marks over each subtree,
/// which puts `w` on exactly the edges of the path.
/// The root's two edges are one unrooted edge.
/// Pairs that meet at the root are one edge closer, and both root edges end up with that edge's
/// sum.
fn weighted_distances_across_edges(ancestry: &Ancestry, d: &ArrayView2<f64>) -> (Vec<f64>, f64) {
    let n = ancestry.len() + 1;
    let root = 2 * n - 2;
    // pauplin_weight[tau] = 2^(1 - tau): the BME weight of a pair of leaves tau edges apart
    let pauplin_weight: Vec<f64> = (0..=2 * n as i32).map(|t| 2f64.powi(1 - t)).collect();

    // Each subtree keeps its leaves, with their distance in edges to the subtree's top node
    let mut leaves: Vec<Vec<(usize, usize)>> = (0..=root)
        .map(|u| if u < n { vec![(u, 0)] } else { Vec::new() })
        .collect();
    // across[u]: the weighted distance across the edge above node u.
    // Until the final pass, it only holds the marks left at u.
    let mut across = vec![0.0; root + 1];
    // total: the BME length, i.e. the weighted distance summed over all pairs
    let mut total = 0.0;

    // Rows list children before parents, so c1 and c2 are finished subtrees when a is reached
    for &[c1, c2, a] in ancestry {
        // Move the children's lists out: each leaf stays in exactly one list, without copies
        let left = std::mem::take(&mut leaves[c1]);
        let right = std::mem::take(&mut leaves[c2]);
        // Edges from c1 and from c2 up to a, counted once for the root's two edges
        let hops = if a == root { 1 } else { 2 };

        // Every pair (x from c1, y from c2) has a as its lowest common ancestor.
        // Its weight w must end up on every edge of its path x ... a ... y.
        // Walking each path would cost its length, so each pair only leaves three marks:
        // +w at x, +w at y and -2w at a.
        // The final pass below gives the edge above each node u the sum of the marks in u's
        // subtree:
        // - an edge on the path from x (or y) up to a has x (or y) below it but not a: +w
        // - an edge above a has x, y and a below it: +w + w - 2w = 0
        // - any other edge has none of them below it: 0
        // This is a difference array on a tree: O(1) work per pair instead of O(path length).
        let mut through_a = 0.0;
        for &(x, dx) in &left {
            let mut from_x = 0.0;
            for &(y, dy) in &right {
                // tau: number of edges on the path x ... a ... y
                let tau = dx + dy + hops;
                // D is assumed symmetric, so D_xy alone is enough
                let w = pauplin_weight[tau] * d[[x, y]];
                // Mark +w at y
                across[y] += w;
                from_x += w;
            }
            // Mark +w at x, once for all its pairs
            across[x] += from_x;
            through_a += from_x;
        }
        // Mark -2w at a, for all pairs that meet here
        across[a] -= 2.0 * through_a;
        total += through_a;

        // a's list: both children's leaves, now one edge further from their top node
        let mut merged = left;
        merged.extend(right);
        merged.iter_mut().for_each(|(_, depth)| *depth += 1);
        leaves[a] = merged;
    }

    // across[u] becomes the sum of the marks in u's subtree.
    // That is the weight of every pair with one leaf below u and the other outside,
    // i.e. of the pairs that the edge above u separates.
    for &[c1, c2, a] in ancestry {
        across[a] += across[c1] + across[c2];
    }
    (across, total)
}

/// Balanced minimum evolution (BME) length of the unrooted tree encoded by a Phylo2Vec vector.
///
/// `L(T) = sum_{i < j} 2^(1 - tau_ij) * D_ij`,
/// where `tau_ij` is the number of edges between leaves `i` and `j` in the unrooted tree
/// (Pauplin, 2000).
/// Lower is better.
/// `D` must be a symmetric `n x n` distance matrix indexed by leaf label.
/// Runs in O(n^2) time and O(n) extra memory.
///
/// # Panics
///
/// Panics if `D` is not `n x n`, where `n = v.len() + 1`.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::vector::objectives::bme_objective;
///
/// // Unrooted tree with 3 leaves: every pair is 2 edges apart
/// let v = vec![0, 0];
/// let d = array![[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]];
/// assert_eq!(bme_objective(&v, &d.view()), 0.5 * (1.0 + 2.0 + 2.0));
/// ```
pub fn bme_objective(v: &[usize], d: &ArrayView2<f64>) -> f64 {
    check_distances(v, d);
    if v.is_empty() {
        return 0.0;
    }
    weighted_distances_across_edges(&to_ancestry(v), d).1
}

/// FastME/Pauplin balanced minimum evolution (BME) branch lengths
/// for the topology encoded by a Phylo2Vec vector.
///
/// Output has the same layout as the branch lengths of a Phylo2Vec matrix:
/// one row per internal node `n + i`,
/// holding the lengths of the edges to its two children in ancestry order.
/// BME is unrooted, so the edge through the root gets one length,
/// split equally between the root's two child edges.
/// The lengths sum to [`bme_objective`].
/// They are returned unclamped and can be negative, unlike Phylo2Vec matrix branch lengths.
/// `D` must be a symmetric `n x n` distance matrix indexed by leaf label.
/// Runs in O(n^2) time and O(n) extra memory.
///
/// # Panics
///
/// Panics if `D` is not `n x n`, where `n = v.len() + 1`.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use phylo2vec::vector::objectives::{bme_branch_lengths, bme_objective};
///
/// let v = vec![0, 0];
/// let d = array![[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]];
/// let bls = bme_branch_lengths(&v, &d.view());
/// let total: f64 = bls.iter().flatten().sum();
/// assert!((total - bme_objective(&v, &d.view())).abs() < 1e-12);
/// ```
pub fn bme_branch_lengths(v: &[usize], d: &ArrayView2<f64>) -> Vec<[f64; 2]> {
    check_distances(v, d);
    if v.is_empty() {
        return Vec::new();
    }
    let ancestry = to_ancestry(v);
    let (across, _) = weighted_distances_across_edges(&ancestry, d);
    let n = v.len() + 1;
    let root = 2 * n - 2;

    // Desper & Gascuel (2002) give the BME length of an edge e = (u, w) from balanced averages
    // between the subtrees around e. Regrouped by leaf pair, each pair (x, y) adds
    // w_xy = 2^(1 - tau_xy) * D_xy to l(e) with a sign:
    // - +w_xy if e is on the path from x to y (together, these pairs give S(e), stored in across)
    // - -w_xy if e hangs off the path, i.e. the path passes through u (or w) without using e
    // - 0 otherwise
    // Let T(u) be the weight of the pairs whose path passes through u.
    // Those that also use e weigh S(e), so those off which e hangs at u weigh T(u) - S(e). Hence
    //     l(e) = S(e) - (T(u) - S(e)) - (T(w) - S(e)) = 3 S(e) - T(u) - T(w)
    // No path passes through a leaf, so T is 0 at leaves.

    // t[u] = T(u). A path through u uses exactly two of u's three edges,
    // so T(u) is half the sum of `across` over them.
    // Row [c1, c2, p] holds all three edges of p (across[c1], across[c2] and across[p]),
    // so one pass fills t without looking up other rows.
    // The root is skipped: in the unrooted tree it lies inside an edge and is not a node.
    let mut t = vec![0.0; root + 1];
    for &[c1, c2, p] in &ancestry {
        if p != root {
            t[p] = 0.5 * (across[c1] + across[c2] + across[p]);
        }
    }

    // l(e) for the edge e = (a, b), where s = S(e):
    // s, minus T(a) - s if a is internal, minus T(b) - s if b is internal.
    // Internal edge: 3 s - T(a) - T(b); pendant edge to leaf a: 2 s - T(b); two-leaf tree: s.
    // t is 0 at leaves, so subtracting t[a] and t[b] unconditionally is safe.
    let edge_length = |a: usize, b: usize, s: f64| {
        (1 + (a >= n) as usize + (b >= n) as usize) as f64 * s - t[a] - t[b]
    };

    ancestry
        .iter()
        .map(|&[c1, c2, p]| {
            if p == root {
                // c1 - root - c2 is one unrooted edge (c1, c2): split its length equally
                let half = 0.5 * edge_length(c1, c2, across[c1]);
                [half, half]
            } else {
                // Edges from p down to its children
                [
                    edge_length(c1, p, across[c1]),
                    edge_length(c2, p, across[c2]),
                ]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::base::sample_vector;
    use crate::vector::graph::{_cophenetic_distances, cophenetic_distances};
    use ndarray::{array, Array2};
    use rand::distr::{Distribution, Uniform};
    use rstest::*;

    fn random_distances(n: usize) -> Array2<f64> {
        let mut rng = rand::rng();
        let unif = Uniform::new(0.1, 1.0).unwrap();
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in (i + 1)..n {
                let x = unif.sample(&mut rng);
                d[[i, j]] = x;
                d[[j, i]] = x;
            }
        }
        d
    }

    fn random_branch_lengths(k: usize) -> Vec<[f64; 2]> {
        let mut rng = rand::rng();
        let unif = Uniform::new(0.01, 1.0).unwrap();
        (0..k)
            .map(|_| [unif.sample(&mut rng), unif.sample(&mut rng)])
            .collect()
    }

    /// Distances for the 3-leaf trees below
    fn three_leaf_distances() -> Array2<f64> {
        array![[0.0, 1.0, 2.0], [1.0, 0.0, 2.5], [2.0, 2.5, 0.0]]
    }

    /// Distances for the 4-leaf trees below
    fn four_leaf_distances() -> Array2<f64> {
        array![
            [0.0, 0.3, 0.8, 0.9],
            [0.3, 0.0, 0.7, 0.6],
            [0.8, 0.7, 0.0, 0.4],
            [0.9, 0.6, 0.4, 0.0]
        ]
    }

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs())),
            "{a} != {b}"
        );
    }

    fn assert_all_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (&x, &y) in a.iter().zip(b) {
            assert_close(x, y);
        }
    }

    // The brute forces take tau from `cophenetic_distances(v, true)`,
    // which counts 2 edges between the leaves of a 2-leaf tree (BME counts 1).
    // They are only valid from 3 leaves.

    /// Brute force: Pauplin's formula with unrooted topological distances.
    fn brute_force_bme(v: &[usize], d: &Array2<f64>) -> f64 {
        let tau = cophenetic_distances(v, true);
        let n = v.len() + 1;
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += 2f64.powf(1.0 - tau[[i, j]]) * d[[i, j]];
            }
        }
        total
    }

    /// Brute force: for each node u, the sum of 2^(1 - tau_xy) * D_xy over the pairs
    /// with x below u and y outside, i.e. the pairs that the edge above u separates.
    fn brute_force_across(v: &[usize], d: &Array2<f64>) -> Vec<f64> {
        let n = v.len() + 1;
        let tau = cophenetic_distances(v, true);
        // below[u][x]: whether leaf x is below node u
        let mut below: Vec<Vec<bool>> = (0..2 * n - 1)
            .map(|u| (0..n).map(|x| x == u).collect())
            .collect();
        for [c1, c2, p] in to_ancestry(v) {
            below[p] = (0..n).map(|x| below[c1][x] || below[c2][x]).collect();
        }
        below
            .iter()
            .map(|inside| {
                let mut s = 0.0;
                for x in (0..n).filter(|&x| inside[x]) {
                    for y in (0..n).filter(|&y| !inside[y]) {
                        s += 2f64.powf(1.0 - tau[[x, y]]) * d[[x, y]];
                    }
                }
                s
            })
            .collect()
    }

    // check_distances

    #[rstest]
    #[case(vec![], 1)]
    #[case(vec![0], 2)]
    #[case(vec![0, 2, 2], 4)]
    fn test_check_distances_matching_shape(#[case] v: Vec<usize>, #[case] n: usize) {
        check_distances(&v, &Array2::<f64>::zeros((n, n)).view());
    }

    #[rstest]
    #[case(vec![], (0, 0))]
    #[case(vec![0, 2, 2], (3, 3))]
    #[case(vec![0, 2, 2], (5, 5))]
    #[case(vec![0, 2, 2], (4, 3))]
    #[case(vec![0, 2, 2], (3, 4))]
    #[should_panic(expected = "Distance matrix must be")]
    fn test_check_distances_wrong_shape(#[case] v: Vec<usize>, #[case] shape: (usize, usize)) {
        check_distances(&v, &Array2::<f64>::zeros(shape).view());
    }

    // weighted_distances_across_edges

    #[test]
    fn test_weighted_distances_null_trees() {
        // 1 leaf: no pairs, no edges
        let (across, total) = weighted_distances_across_edges(&vec![], &array![[0.0]].view());
        assert_eq!((across, total), (vec![0.0], 0.0));

        // 2 leaves: one edge, so tau = 1; both root children hold its sum
        let d = array![[0.0, 0.7], [0.7, 0.0]];
        let (across, total) = weighted_distances_across_edges(&to_ancestry(&[0]), &d.view());
        assert_all_close(&across[..2], &[0.7, 0.7]);
        assert_close(total, 0.7);
    }

    #[test]
    fn test_weighted_distances_three_leaves() {
        // Star: every pair is 2 edges apart, so each pendant edge holds half of its leaf's row
        let v = vec![0, 0];
        let d = three_leaf_distances();
        let ancestry = to_ancestry(&v);
        // Cherry (0, 2) at node 3; the root joins 3 and leaf 1, so edge 3-1 is leaf 1's
        assert_eq!(ancestry, vec![[0, 2, 3], [3, 1, 4]]);
        let (across, total) = weighted_distances_across_edges(&ancestry, &d.view());
        let half_row = |x: usize| 0.5 * d.row(x).sum();
        assert_all_close(
            &across[..4],
            &[half_row(0), half_row(1), half_row(2), half_row(1)],
        );
        assert_close(total, 0.5 * (d[[0, 1]] + d[[0, 2]] + d[[1, 2]]));
    }

    #[test]
    fn test_weighted_distances_four_leaves() {
        // ((0,1),(2,3)): pairs within a cherry are 2 edges apart, the others 3
        let v = vec![0, 2, 2];
        let d = four_leaf_distances();
        let ancestry = to_ancestry(&v);
        assert_eq!(ancestry, vec![[2, 3, 4], [0, 1, 5], [5, 4, 6]]);
        let (across, total) = weighted_distances_across_edges(&ancestry, &d.view());
        // Internal edge 5-4 (through the root): the four pairs across the cherries
        let s_e = 0.25 * (d[[0, 2]] + d[[0, 3]] + d[[1, 2]] + d[[1, 3]]);
        let expected = [
            0.5 * d[[0, 1]] + 0.25 * (d[[0, 2]] + d[[0, 3]]),
            0.5 * d[[0, 1]] + 0.25 * (d[[1, 2]] + d[[1, 3]]),
            0.5 * d[[2, 3]] + 0.25 * (d[[0, 2]] + d[[1, 2]]),
            0.5 * d[[2, 3]] + 0.25 * (d[[0, 3]] + d[[1, 3]]),
            s_e,
            s_e,
        ];
        assert_all_close(&across[..6], &expected);
        assert_close(total, 0.5 * (d[[0, 1]] + d[[2, 3]]) + s_e);
    }

    #[rstest]
    #[case(3)]
    #[case(4)]
    #[case(10)]
    #[case(50)]
    #[case(200)]
    fn test_weighted_distances_match_brute_force(#[case] n_leaves: usize) {
        for ordered in [true, false] {
            let v = sample_vector(n_leaves, ordered);
            let d = random_distances(n_leaves);
            let (across, total) = weighted_distances_across_edges(&to_ancestry(&v), &d.view());
            let root = 2 * n_leaves - 2;
            // across[root] is unused
            assert_all_close(&across[..root], &brute_force_across(&v, &d)[..root]);
            assert_close(total, brute_force_bme(&v, &d));
        }
    }

    #[test]
    fn test_weighted_distances_caterpillar() {
        // Deepest tree: leaf lists and tau reach their maximum
        let n_leaves = 50;
        let v: Vec<usize> = (0..n_leaves - 1).collect();
        let d = random_distances(n_leaves);
        let (across, total) = weighted_distances_across_edges(&to_ancestry(&v), &d.view());
        let root = 2 * n_leaves - 2;
        assert_all_close(&across[..root], &brute_force_across(&v, &d)[..root]);
        assert_close(total, brute_force_bme(&v, &d));
    }

    #[test]
    fn test_weighted_distances_root_invariance() {
        // [0, 2, 2] and [0, 1, 2] are ((0,1),(2,3)) rooted on the internal edge
        // and on leaf 0's edge: the same unrooted tree
        let d = four_leaf_distances();
        let balanced = to_ancestry(&[0, 2, 2]);
        let ladder = to_ancestry(&[0, 1, 2]);
        assert_eq!(balanced, vec![[2, 3, 4], [0, 1, 5], [5, 4, 6]]);
        assert_eq!(ladder, vec![[2, 3, 4], [1, 4, 5], [0, 5, 6]]);
        let (across_balanced, total_balanced) =
            weighted_distances_across_edges(&balanced, &d.view());
        let (across_ladder, total_ladder) = weighted_distances_across_edges(&ladder, &d.view());
        // Nodes 0 to 4 are the same in both: the leaves, and the cherry (2, 3),
        // whose edge above is the internal edge
        assert_all_close(&across_balanced[..5], &across_ladder[..5]);
        assert_close(total_balanced, total_ladder);
    }

    // bme_objective

    #[test]
    fn test_bme_objective_null_trees() {
        assert_eq!(bme_objective(&[], &array![[0.0]].view()), 0.0);
        let d = array![[0.0, 0.7], [0.7, 0.0]];
        assert_close(bme_objective(&[0], &d.view()), 0.7);
    }

    #[test]
    fn test_bme_objective_three_leaves() {
        // Star: every pair is 2 edges apart
        let d = three_leaf_distances();
        assert_close(
            bme_objective(&[0, 0], &d.view()),
            0.5 * (d[[0, 1]] + d[[0, 2]] + d[[1, 2]]),
        );
    }

    #[test]
    fn test_bme_objective_four_leaves() {
        // ((0,1),(2,3)): pairs within a cherry are 2 edges apart, the others 3
        let d = four_leaf_distances();
        let expected =
            0.5 * (d[[0, 1]] + d[[2, 3]]) + 0.25 * (d[[0, 2]] + d[[0, 3]] + d[[1, 2]] + d[[1, 3]]);
        assert_close(bme_objective(&[0, 2, 2], &d.view()), expected);
    }

    #[rstest]
    #[case(3)]
    #[case(4)]
    #[case(10)]
    #[case(50)]
    #[case(200)]
    fn test_bme_objective_matches_brute_force(#[case] n_leaves: usize) {
        for ordered in [true, false] {
            let v = sample_vector(n_leaves, ordered);
            let d = random_distances(n_leaves);
            assert_close(bme_objective(&v, &d.view()), brute_force_bme(&v, &d));
        }
    }

    #[test]
    fn test_bme_objective_root_invariance() {
        // [0, 2, 2] and [0, 1, 2] are the same unrooted tree, rooted differently
        let d = four_leaf_distances();
        assert_close(
            bme_objective(&[0, 2, 2], &d.view()),
            bme_objective(&[0, 1, 2], &d.view()),
        );
    }

    #[test]
    #[should_panic(expected = "Distance matrix must be 4 x 4")]
    fn test_bme_objective_wrong_shape() {
        bme_objective(&[0, 2, 2], &Array2::<f64>::zeros((3, 3)).view());
    }

    // bme_branch_lengths

    #[test]
    fn test_bme_branch_lengths_null_trees() {
        assert!(bme_branch_lengths(&[], &array![[0.0]].view()).is_empty());
        // 2 leaves: the one edge gets D01, split between the root's two child edges
        let d = array![[0.0, 0.7], [0.7, 0.0]];
        assert_eq!(bme_branch_lengths(&[0], &d.view()), vec![[0.35, 0.35]]);
    }

    #[test]
    fn test_bme_branch_lengths_three_leaves() {
        // Star: pendant edge of x is 0.5 * (D_xy + D_xz - D_yz)
        let v = vec![0, 0];
        let d = three_leaf_distances();
        assert_eq!(to_ancestry(&v), vec![[0, 2, 3], [3, 1, 4]]);
        let pendant = |x: usize, y: usize, z: usize| 0.5 * (d[[x, y]] + d[[x, z]] - d[[y, z]]);
        // Edge 3-1 (through the root) is leaf 1's pendant edge
        let expected = [
            [pendant(0, 1, 2), pendant(2, 0, 1)],
            [0.5 * pendant(1, 0, 2), 0.5 * pendant(1, 0, 2)],
        ];
        assert_all_close(
            &bme_branch_lengths(&v, &d.view()).concat(),
            &expected.concat(),
        );
    }

    #[test]
    fn test_bme_branch_lengths_four_leaves() {
        // ((0,1),(2,3)), Desper & Gascuel's closed forms:
        // internal edge 0.25 * (D02 + D03 + D12 + D13) - 0.5 * (D01 + D23);
        // pendant edge of x, with sibling y and other leaves z, w:
        // 0.5 * (D_xy + 0.5 * (D_xz + D_xw) - 0.5 * (D_yz + D_yw))
        let v = vec![0, 2, 2];
        let d = four_leaf_distances();
        assert_eq!(to_ancestry(&v), vec![[2, 3, 4], [0, 1, 5], [5, 4, 6]]);
        let pendant = |x: usize, y: usize, z: usize, w: usize| {
            0.5 * (d[[x, y]] + 0.5 * (d[[x, z]] + d[[x, w]]) - 0.5 * (d[[y, z]] + d[[y, w]]))
        };
        let internal =
            0.25 * (d[[0, 2]] + d[[0, 3]] + d[[1, 2]] + d[[1, 3]]) - 0.5 * (d[[0, 1]] + d[[2, 3]]);
        let expected = [
            [pendant(2, 3, 0, 1), pendant(3, 2, 0, 1)],
            [pendant(0, 1, 2, 3), pendant(1, 0, 2, 3)],
            [0.5 * internal, 0.5 * internal],
        ];
        assert_all_close(
            &bme_branch_lengths(&v, &d.view()).concat(),
            &expected.concat(),
        );
    }

    #[rstest]
    #[case(3)]
    #[case(4)]
    #[case(10)]
    #[case(100)]
    fn test_bme_branch_lengths_sum_to_bme_length(#[case] n_leaves: usize) {
        for ordered in [true, false] {
            let v = sample_vector(n_leaves, ordered);
            let d = random_distances(n_leaves);
            let total: f64 = bme_branch_lengths(&v, &d.view()).iter().flatten().sum();
            assert_close(total, bme_objective(&v, &d.view()));
        }
    }

    #[rstest]
    #[case(3)]
    #[case(4)]
    #[case(10)]
    #[case(100)]
    fn test_bme_branch_lengths_recover_additive_distances(#[case] n_leaves: usize) {
        // On a tree metric, BME branch lengths are exact; only the split of the root edge
        // between the two root children is a convention.
        for ordered in [true, false] {
            let v = sample_vector(n_leaves, ordered);
            let true_bls = random_branch_lengths(v.len());
            let d = _cophenetic_distances(&v, Some(&true_bls), false);
            let bls = bme_branch_lengths(&v, &d.view());
            let k = v.len();
            for i in 0..k - 1 {
                assert_close(bls[i][0], true_bls[i][0]);
                assert_close(bls[i][1], true_bls[i][1]);
            }
            let root_edge = true_bls[k - 1][0] + true_bls[k - 1][1];
            assert_close(bls[k - 1][0], 0.5 * root_edge);
            assert_close(bls[k - 1][1], 0.5 * root_edge);
        }
    }

    #[test]
    fn test_bme_branch_lengths_root_invariance() {
        // [0, 2, 2] and [0, 1, 2] are ((0,1),(2,3)) rooted on the internal edge
        // and on leaf 0's edge: each unrooted edge must get the same length
        let d = four_leaf_distances();
        // Rows [2, 3, 4], [0, 1, 5], [5, 4, 6]
        let balanced = bme_branch_lengths(&[0, 2, 2], &d.view());
        // Rows [2, 3, 4], [1, 4, 5], [0, 5, 6]
        let ladder = bme_branch_lengths(&[0, 1, 2], &d.view());
        // Pendant edges of leaves 2 and 3
        assert_close(balanced[0][0], ladder[0][0]);
        assert_close(balanced[0][1], ladder[0][1]);
        // Pendant edge of leaf 0 (through the root in ladder)
        assert_close(balanced[1][0], ladder[2][0] + ladder[2][1]);
        // Pendant edge of leaf 1
        assert_close(balanced[1][1], ladder[1][0]);
        // Internal edge (through the root in balanced)
        assert_close(balanced[2][0] + balanced[2][1], ladder[1][1]);
    }

    #[test]
    fn test_bme_branch_lengths_negative() {
        // D fits ((0,2),(1,3)) better than ((0,1),(2,3)),
        // so the internal edge of ((0,1),(2,3)) is negative: returned as is, not clamped
        let d = array![
            [0.0, 1.0, 0.6, 0.6],
            [1.0, 0.0, 0.6, 0.6],
            [0.6, 0.6, 0.0, 1.0],
            [0.6, 0.6, 1.0, 0.0]
        ];
        let bls = bme_branch_lengths(&[0, 2, 2], &d.view());
        // Internal edge: 0.25 * 4 * 0.6 - 0.5 * (1.0 + 1.0) = -0.4, split over the root row
        assert_all_close(&bls[2], &[-0.2, -0.2]);
    }

    #[test]
    #[should_panic(expected = "Distance matrix must be 4 x 4")]
    fn test_bme_branch_lengths_wrong_shape() {
        bme_branch_lengths(&[0, 2, 2], &Array2::<f64>::zeros((3, 3)).view());
    }
}
