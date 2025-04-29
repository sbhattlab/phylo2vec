/// A type alias for the Pair type, which is a tuple representing (child1, child2)
pub type Pair = (usize, usize);

/// A type alias for the Ancestry type, which is a vector of vectors representing [child1, child2, parent]
pub type Ancestry = Vec<[usize; 3]>;

/// A type alias for the PairsVec type, which is a vector of tuples representing (child1, child2)
pub type Pairs = Vec<Pair>;
