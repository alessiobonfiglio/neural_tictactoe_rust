use std::collections::HashMap;
use std::mem;
use std::time::Instant;

use byte_unit::Byte;
use nalgebra::{SMatrix, SVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::tictactoe_minmax::{
    Minmax, TICTACTOE_GRID_SIZE, TICTACTOE_SIZE, TicTacToeCell, TicTacToePlayer, TicTacToeState,
};
use crate::tictactoe_nn_solver::TicTacToeNNSolver;
use crate::TicTacToeCell::{Assigned, Empty};

mod tictactoe_minmax;
mod tictactoe_nn_solver;

//const BATCH_SIZE: usize = 126;
const BATCH_SIZE: usize = 1;
const HIDDEN_LAYER_SIZE: usize = 20;

fn main() {
    println!("Welcome to Neural TicTacToe (made in rust)\n");
    let mut rng = StdRng::seed_from_u64(222);

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
    assert_eq!(
        dataset.len() % BATCH_SIZE,
        0,
        "The batch size ({}) must be a factor od the dataset size ({})",
        BATCH_SIZE,
        dataset.len()
    );
    println!(
        "Dataset size: {} elems ({})\n",
        dataset.len(),
        Byte::from_bytes(mem::size_of_val(&dataset[..]) as u128).get_appropriate_unit(true)
    );

    println!("Constructing Neural Network Model...");
    let start = Instant::now();
    let network = TicTacToeNNSolver::<HIDDEN_LAYER_SIZE>::new(&mut rng);
    let duration = start.elapsed();
    println!(
        "Time elapsed for constructing Neural Network Model: {:?}",
        duration
    );
}

fn convert_minmax_results_to_dataset(
    mut map: HashMap<TicTacToeState, SMatrix<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>>,
) -> Vec<(
    SVector<f32, TICTACTOE_GRID_SIZE>,
    SVector<f32, TICTACTOE_GRID_SIZE>,
)> {
    let mut ret = Vec::new();
    for (k, v) in map.drain() {
        let mut x = SVector::zeros();
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
        let max_value = *v.iter().max().unwrap();
        // assign y to 1 only when it's a valid action and has the maximum value for
        // this state
        let mut y = SVector::repeat(1.);
        for i in 0..TICTACTOE_SIZE {
            for j in 0..TICTACTOE_SIZE {
                let ii = i * TICTACTOE_SIZE + j;
                if k[i][j] != Empty || v[(i, j)] != max_value {
                    y[ii] = 0.;
                }
            }
        }

        ret.push((x, y));
    }

    ret
}
