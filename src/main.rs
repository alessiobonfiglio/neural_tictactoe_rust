use std::collections::HashMap;
use std::mem;
use std::time::Instant;

use byte_unit::Byte;

use crate::tictactoe_minmax::{
    Minmax, TICTACTOE_SIZE, TicTacToeCell, TicTacToePlayer, TicTacToeState,
};
use crate::TicTacToeCell::{Assigned, Empty};

mod tictactoe_minmax;

fn main() {
    println!("Welcome to Neural TicTacToe (made in rust)\n");

    println!("Generating Dataset from Minmax...");
    let start = Instant::now();
    let mut minmax_results = Minmax::execute(TicTacToePlayer::X);
    minmax_results.extend(Minmax::execute(TicTacToePlayer::O));
    let duration = start.elapsed();
    println!("Time elapsed for Minmax: {:?}", duration);
    let start = Instant::now();
    // convert the dataset in a more suitable format (Vec<(input, output)>:
    // input  -> 0: empty cell
    //           1: X
    //          -1: O
    // output -> 0: suboptimal/unfeasible move
    //           1: optimal move
    let dataset = convert_minmax_results_to_dataset(minmax_results);
    let duration = start.elapsed();
    println!("Time elapsed for converting the Dataset: {:?}", duration);
    println!(
        "Dataset size: {} elems ({})\n",
        dataset.len(),
        Byte::from_bytes(mem::size_of_val(&dataset[..]) as u128).get_appropriate_unit(true)
    );
}

fn convert_minmax_results_to_dataset(
    mut map: HashMap<TicTacToeState, [[i32; TICTACTOE_SIZE]; TICTACTOE_SIZE]>,
) -> Vec<(
    [f32; TICTACTOE_SIZE * TICTACTOE_SIZE],
    [f32; TICTACTOE_SIZE * TICTACTOE_SIZE],
)> {
    let mut ret = Vec::new();
    for (k, v) in map.drain() {
        let mut x = [0.; TICTACTOE_SIZE * TICTACTOE_SIZE];
        for i in 0..TICTACTOE_SIZE {
            for j in 0..TICTACTOE_SIZE {
                let ii = i * TICTACTOE_SIZE + j;
                x[ii] = match k[i][j] {
                    Empty => 0.,
                    Assigned(TicTacToePlayer::X) => 1.,
                    Assigned(TicTacToePlayer::O) => -1.,
                };
            }
        }
        let max_value = *v.iter().flatten().max().unwrap();
        // assign y to 1 only when it's a valid action and has the maximum value for
        // this state
        let mut y = [1.; TICTACTOE_SIZE * TICTACTOE_SIZE];
        for i in 0..TICTACTOE_SIZE {
            for j in 0..TICTACTOE_SIZE {
                let ii = i * TICTACTOE_SIZE + j;
                if k[i][j] != Empty || v[i][j] != max_value {
                    y[ii] = 0.;
                }
            }
        }

        ret.push((x, y));
    }

    ret
}
