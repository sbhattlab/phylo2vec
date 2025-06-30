use std::cmp::Ordering;

use crate::types::Pair;

pub struct Node {
    value: Pair,
    height: usize,
    size: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn new(value: Pair) -> Self {
        Node {
            value,
            height: 1,
            size: 1,
            left: None,
            right: None,
        }
    }
}

/// An AVL tree is a self-balancing binary search tree.
pub struct AVLTree {
    root: Option<Box<Node>>,
}

impl AVLTree {
    pub fn new() -> Self {
        AVLTree { root: None }
    }

    pub fn with_vector(v: &[usize]) -> Self {
        let mut avl_tree = AVLTree::default();
        let k = v.len();
        avl_tree.insert(0, (0, 1));

        for (i, &vi) in v.iter().enumerate().take(k).skip(1) {
            let next_leaf = i + 1;
            if vi <= i {
                avl_tree.insert(0, (v[i], next_leaf));
            } else {
                let index = v[i] - next_leaf;
                let pair = AVLTree::lookup(&avl_tree, index);
                avl_tree.insert(index + 1, (pair.0, next_leaf));
            }
        }

        avl_tree
    }

    fn get_height(node: &Option<Box<Node>>) -> usize {
        match node {
            Some(ref n) => n.height,
            None => 0,
        }
    }

    fn get_size(node: &Option<Box<Node>>) -> usize {
        match node {
            Some(ref n) => n.size,
            None => 0,
        }
    }

    fn update_height_and_size(n: &mut Node) {
        n.height = 1 + usize::max(Self::get_height(&n.left), Self::get_height(&n.right));
        n.size = 1 + Self::get_size(&n.left) + Self::get_size(&n.right);
    }

    fn right_rotate(y: &mut Option<Box<Node>>) -> Option<Box<Node>> {
        if let Some(mut y_node) = y.take() {
            if let Some(mut x) = y_node.left.take() {
                // Perform rotation
                let t2 = x.right.take();
                x.right = Some(y_node);
                x.right.as_mut().unwrap().left = t2;

                // Update height and size values
                Self::update_height_and_size(x.right.as_mut().unwrap());
                Self::update_height_and_size(&mut x);

                Some(x)
            } else {
                // If no left child, revert the state and return `None`
                *y = Some(y_node);
                None
            }
        } else {
            None
        }
    }

    fn left_rotate(x: &mut Option<Box<Node>>) -> Option<Box<Node>> {
        if let Some(mut x_node) = x.take() {
            if let Some(mut y) = x_node.right.take() {
                // Perform rotation
                let t2 = y.left.take();
                y.left = Some(x_node);
                y.left.as_mut().unwrap().right = t2;

                // Update height and size values
                Self::update_height_and_size(y.left.as_mut().unwrap());
                Self::update_height_and_size(&mut y);

                Some(y)
            } else {
                // If no right child, revert the state and return `None`
                *x = Some(x_node);
                None
            }
        } else {
            None
        }
    }

    fn get_balance_factor(node: &Option<Box<Node>>) -> isize {
        // Balance factor is the difference between the height of the left subtree and the right subtree.
        match node {
            Some(ref n) => Self::get_height(&n.left) as isize - Self::get_height(&n.right) as isize,
            None => 0,
        }
    }

    fn balance(node: &mut Option<Box<Node>>) -> Option<Box<Node>> {
        let balance_factor = Self::get_balance_factor(node);
        if balance_factor > 1 {
            if Self::get_balance_factor(&node.as_ref().unwrap().left) >= 0 {
                return Self::right_rotate(node);
            } else {
                if let Some(ref mut n) = node {
                    n.left = Self::left_rotate(&mut n.left);
                }
                return Self::right_rotate(node);
            }
        }
        if balance_factor < -1 {
            if Self::get_balance_factor(&node.as_ref().unwrap().right) <= 0 {
                return Self::left_rotate(node);
            } else {
                if let Some(ref mut n) = node {
                    n.right = Self::right_rotate(&mut n.right);
                }
                return Self::left_rotate(node);
            }
        }
        // An AVL tree is balanced if its balance factor is -1, 0, or 1.
        node.take()
    }

    pub fn insert(&mut self, index: usize, value: Pair) {
        self.root = Self::insert_by_index(self.root.take(), value, index);
    }

    fn insert_by_index(node: Option<Box<Node>>, value: Pair, index: usize) -> Option<Box<Node>> {
        let mut n: Box<Node> = match node {
            Some(n) => n,
            None => return Some(Box::new(Node::new(value))),
        };

        let left_size = Self::get_size(&n.left);
        if index <= left_size {
            n.left = Self::insert_by_index(n.left.take(), value, index);
        } else {
            n.right = Self::insert_by_index(n.right.take(), value, index - left_size - 1);
        }

        Self::update_height_and_size(&mut n);

        Self::balance(&mut Some(n))
    }

    pub fn lookup(&self, index: usize) -> Pair {
        Self::lookup_node(&self.root, index).unwrap_or((0, 0))
    }

    fn lookup_node(node: &Option<Box<Node>>, index: usize) -> Option<Pair> {
        match node {
            Some(ref n) => {
                let left_size = Self::get_size(&n.left);
                match index.cmp(&left_size) {
                    Ordering::Less => Self::lookup_node(&n.left, index),
                    Ordering::Equal => Some(n.value),
                    Ordering::Greater => Self::lookup_node(&n.right, index - left_size - 1),
                }
            }
            None => None,
        }
    }

    pub fn inorder_traversal(&self) -> Vec<Pair> {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        let mut current = &self.root;

        while current.is_some() || !stack.is_empty() {
            while let Some(ref n) = current {
                stack.push(n);
                current = &n.left;
            }

            let node = stack.pop().unwrap();
            result.push(node.value);

            current = &node.right;
        }

        result
    }
}

impl Default for AVLTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn sample_tree() -> AVLTree {
        let mut tree = AVLTree::default();
        tree.insert(0, (1, 1));
        tree.insert(1, (2, 2));
        tree.insert(2, (3, 3));
        tree
    }

    #[rstest]
    #[case(0, (1, 1))]
    #[case(1, (2, 2))]
    #[case(2, (3, 3))]
    fn test_lookup(#[case] lookup_index: usize, #[case] expected: Pair) {
        let tree = sample_tree();
        assert_eq!(tree.lookup(lookup_index), expected);
    }

    #[rstest]
    #[case(vec![(0, (1, 1))], 0, (1, 1))]
    #[case(vec![(0, (1, 1)), (1, (2, 2))], 1, (2, 2))]
    #[case(vec![(0, (1, 1)), (0, (2, 2)), (0, (3, 3))], 0, (3, 3))]
    #[case(vec![(0, (1, 1)), (0, (2, 2)), (0, (3, 3))], 2, (1, 1))]
    fn test_insert_with_lookup(
        #[case] inserts: Vec<(usize, Pair)>,
        #[case] lookup_index: usize,
        #[case] expected: Pair,
    ) {
        let mut tree = AVLTree::default();
        for (index, value) in inserts {
            tree.insert(index, value);
        }
        assert_eq!(tree.lookup(lookup_index), expected);
    }

    #[rstest]
    #[case(vec![(0, (1, 1)), (1, (2, 2)), (2, (3, 3))], vec![(1, 1), (2, 2), (3, 3)])]
    #[case(vec![(0, (3, 3)), (0, (2, 2)), (0, (1, 1))], vec![(1, 1), (2, 2), (3, 3)])]
    #[case(vec![(0, (2, 2)), (1, (1, 1)), (0, (3, 3))], vec![(3, 3), (2, 2), (1, 1)])]
    fn test_inorder_traversal(#[case] inserts: Vec<(usize, Pair)>, #[case] expected: Vec<Pair>) {
        let mut tree = AVLTree::default();
        for (index, value) in inserts {
            tree.insert(index, value);
        }
        assert_eq!(tree.inorder_traversal(), expected);
    }

    #[rstest]
    fn test_empty_tree() {
        let tree = AVLTree::default();
        assert!(tree.inorder_traversal().is_empty());
    }

    #[rstest]
    #[case((0, (1, 1)), vec![(1, 1)])]
    fn test_single_element_insert(#[case] insert: (usize, Pair), #[case] expected: Vec<Pair>) {
        let mut tree = AVLTree::default();
        tree.insert(insert.0, insert.1);
        assert_eq!(tree.inorder_traversal(), expected);
    }

    #[rstest]
    #[case(vec![(0, (1, 1)), (1, (2, 2))], vec![(1, 1), (2, 2)])]
    #[case(vec![(1, (1, 1)), (0, (2, 2))], vec![(2, 2), (1, 1)])]
    fn test_two_elements_insert(#[case] inserts: Vec<(usize, Pair)>, #[case] expected: Vec<Pair>) {
        let mut tree = AVLTree::default();
        for (index, value) in inserts {
            tree.insert(index, value);
        }
        assert_eq!(tree.inorder_traversal(), expected);
    }

    #[rstest]
    #[case(3, (0, 0))]
    #[case(10, (0, 0))]
    #[case(usize::MAX, (0, 0))]
    fn test_lookup_out_of_bounds(
        sample_tree: AVLTree,
        #[case] index: usize,
        #[case] expected: Pair,
    ) {
        assert_eq!(sample_tree.lookup(index), expected);
    }

    #[rstest]
    #[case(vec![(0, (1, 1)), (1, (1, 1)), (2, (1, 1))], vec![(1, 1), (1, 1), (1, 1)])]
    fn test_insert_duplicates(#[case] inserts: Vec<(usize, Pair)>, #[case] expected: Vec<Pair>) {
        let mut tree = AVLTree::default();
        for (index, value) in inserts {
            tree.insert(index, value);
        }
        assert_eq!(tree.inorder_traversal(), expected);
    }

    #[rstest]
    #[case(vec![0, 1, 2, 3, 4, 5])]
    #[case(vec![5, 4, 3, 2, 1, 0])]
    #[case(vec![3, 1, 4, 0, 2, 5])]
    fn test_balance_after_insert(#[case] insert_order: Vec<usize>) {
        let mut tree = AVLTree::default();
        for (i, &index) in insert_order.iter().enumerate() {
            tree.insert(index, (i, i));
        }
        // After balancing, the height should be significantly less than the number of nodes
        assert!(AVLTree::get_height(&tree.root) <= 4);
    }

    #[rstest]
    #[case(vec![5, 3, 7, 2, 4, 6, 8])]
    fn test_balance_after_insert_granular(#[case] inserts: Vec<usize>) {
        let mut tree = AVLTree::default();

        for &index in inserts.iter() {
            tree.insert(index, (index, index));
        }
        // Check balance factor for every node in the tree
        test_balance_helper(&tree.root);
    }

    fn test_balance_helper(node: &Option<Box<Node>>) {
        if let Some(ref n) = node {
            let balance_factor = AVLTree::get_balance_factor(node);
            assert!(
                (-1..=1).contains(&balance_factor),
                "Node with value {:?} is unbalanced! Balance factor: {}",
                n.value,
                balance_factor
            );

            // Recursively check balance for left and right subtrees
            test_balance_helper(&n.left);
            test_balance_helper(&n.right);
        }
    }
}
