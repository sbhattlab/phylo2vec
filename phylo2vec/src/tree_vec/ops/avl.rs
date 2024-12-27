use crate::tree_vec::types::Pair;

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

    fn update_height(n: &mut Box<Node>) {
        n.height = 1 + usize::max(Self::get_height(&n.left), Self::get_height(&n.right));
    }

    fn update_size(n: &mut Box<Node>) {
        n.size = 1 + Self::get_size(&n.left) + Self::get_size(&n.right);
    }

    fn right_rotate(y: &mut Option<Box<Node>>) -> Option<Box<Node>> {
        if let Some(mut y_node) = y.take() {
            if let Some(mut x) = y_node.left.take() {
                // Perform rotation
                let t2 = x.right.take();
                x.right = Some(y_node);
                x.right.as_mut().unwrap().left = t2;

                // Update heights
                Self::update_height(x.right.as_mut().unwrap());
                Self::update_height(&mut x);

                return Some(x);
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

                // Update heights
                Self::update_height(y.left.as_mut().unwrap());
                Self::update_height(&mut y);

                return Some(y);
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

        Self::update_height(&mut n);
        Self::update_size(&mut n);

        return Self::balance(&mut Some(n));
    }

    pub fn lookup(&self, index: usize) -> Pair {
        Self::lookup_node(&self.root, index).unwrap_or((0, 0))
    }

    fn lookup_node(node: &Option<Box<Node>>, index: usize) -> Option<Pair> {
        match node {
            Some(ref n) => {
                let left_size = Self::get_size(&n.left);
                if index < left_size {
                    Self::lookup_node(&n.left, index)
                } else if index == left_size {
                    Some(n.value)
                } else {
                    Self::lookup_node(&n.right, index - left_size - 1)
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

    pub fn get_pairs(&self) -> Vec<Pair> {
        self.inorder_traversal()
    }
}
