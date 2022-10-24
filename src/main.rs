use std::time::Instant;

use crate::tictactoe_minmax::{Minmax, TicTacToePlayer};

mod tictactoe_minmax;

fn main() {
    println!("Welcome to Neural TicTacToe (made in rust)\n");

    println!("Generating Dataset from Minmax...");
    let start = Instant::now();
    let mut minmax_results = Minmax::execute(TicTacToePlayer::X);
    minmax_results.extend(Minmax::execute(TicTacToePlayer::O));
    let duration = start.elapsed();
    println!("Time elapsed for Minmax: {:?}", duration);
}
