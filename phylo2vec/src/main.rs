mod to_newick;
mod avl;

fn main() {
    // Currently a small testing routine to ensure things are working as intended
    let v = vec![0,2,2,5,2];
    let newick_string = to_newick::to_newick(v);
    print!("{}", newick_string);
}