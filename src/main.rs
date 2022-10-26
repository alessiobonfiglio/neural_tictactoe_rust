use std::collections::HashMap;
use std::mem;
use std::mem::size_of;
use std::time::Instant;

use byte_unit::Byte;
use console::style;
use nalgebra::{SMatrix, SVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

use dialoguer::{Input, Select, theme::ColorfulTheme};

use crate::tictactoe_minmax::{Minmax, TICTACTOE_GRID_SIZE, TICTACTOE_SIZE, TicTacToeCell, TicTacToePlayer, TicTacToeState};
use crate::tictactoe_solver_nn::{TicTacToeSolverNN, TrainStopCondition};
use crate::TicTacToeCell::{Assigned, Empty};
use crate::TrainStopCondition::{Accuracy, Epoch, Loss, Perfect};

mod tictactoe_minmax;
mod tictactoe_solver_nn;

const HIDDEN_LAYER_SIZE: usize = 64;
const BATCH_SIZE: usize = 36;
const LEARNING_RATE: f32 = 0.001;

fn main() {
    println!("Welcome to Neural TicTacToe (made in rust)\n");
    let mut rng = StdRng::seed_from_u64(222);

    println!("{}", style("Generating Dataset from Minmax...").bold());
    let start = Instant::now();
    let mut minmax_results = Minmax::execute(TicTacToePlayer::X);
    minmax_results.extend(Minmax::execute(TicTacToePlayer::O));
    let duration = start.elapsed();
    println!("Time elapsed for Minmax: {:?}", style(duration).green());
    let start = Instant::now();
    // convert the dataset in a more suitable format (Vec<(input, output)>:
    // input  -> 0: empty cell
    //           1: X
    //          -1: O
    // output -> 0: suboptimal/unfeasible move
    //           1: optimal move
    let dataset = convert_minmax_results_to_dataset(minmax_results);
    let duration = start.elapsed();
    println!("Time elapsed for converting the Dataset: {:?}", style(duration).green());
    assert_eq!(
        dataset.0.len() % BATCH_SIZE,
        0,
        "The batch size ({}) must be a factor od the dataset size ({})",
        BATCH_SIZE,
        dataset.0.len()
    );
    println!(
        "Dataset size: {} elems ({})\n",
        style(dataset.0.len()).green(),
        style(Byte::from_bytes(2 * mem::size_of_val(&dataset.0[..]) as u128).get_appropriate_unit(true)).green()
    );

    println!("{}", style("Constructing Neural Network Model...").bold());
    let start = Instant::now();
    let mut network = TicTacToeSolverNN::<HIDDEN_LAYER_SIZE>::new(&mut rng);
    let duration = start.elapsed();
    println!("\t      Input layer            (size: {})", style(TICTACTOE_GRID_SIZE).green());
    println!("\t           |\n\t           V");
    println!(
        "\tHidden dense layer + ReLU    (size: {},\t{} params: {})",
        style(HIDDEN_LAYER_SIZE).green(),
        style(format!(
            "{} + {}",
            network.get_parameters_size().0 / size_of::<f32>(),
            network.get_parameters_size().1 / size_of::<f32>()
        ))
            .green(),
        style(Byte::from_bytes((network.get_parameters_size().0 + network.get_parameters_size().1) as u128).get_appropriate_unit(true)).green()
    );
    println!("\t           |\n\t           V");
    println!(
        "\tOutput layer + Sigmoid       (size: {},\t{} params: {})",
        style(TICTACTOE_GRID_SIZE).green(),
        style(format!(
            "{} + {}",
            network.get_parameters_size().2 / size_of::<f32>(),
            network.get_parameters_size().3 / size_of::<f32>()
        ))
            .green(),
        style(Byte::from_bytes((network.get_parameters_size().2 + network.get_parameters_size().3) as u128).get_appropriate_unit(true)).green()
    );
    println!("Time elapsed for constructing Neural Network Model: {:?}\n", style(duration).green());

    let selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select how to train the network")
        .item("Until it learns the task perfectly")
        .item("For a certain number of epochs at most")
        .item("Until it reaches a certain accuracy")
        .item("Until it reaches a certain loss")
        .default(0)
        .interact()
        .unwrap();
    let train_stop_condition = match selection {
        0 => Perfect,
        1 => {
            let e = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Enter the number of epochs")
                .validate_with(|input: &String| -> Result<(), &str> {
                    if let Ok(n) = input.parse::<u32>() {
                        if n > 0 {
                            Ok(())
                        } else {
                            Err("Must be greater than 0")
                        }
                    } else {
                        Err("Must be a integer number greater than 0")
                    }
                })
                .interact_text()
                .unwrap();
            Epoch(e.parse::<u32>().unwrap())
        }
        2 => {
            let a = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Enter the target accuracy")
                .validate_with(|input: &String| -> Result<(), &str> {
                    if let Ok(n) = input.parse::<f32>() {
                        if (0.0..=1.0).contains(&n) {
                            Ok(())
                        } else {
                            Err("Must be between 0.0 an 1.0")
                        }
                    } else {
                        Err("Must be a number")
                    }
                })
                .interact_text()
                .unwrap();
            Accuracy(a.parse::<f32>().unwrap())
        }
        3 => {
            let l = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Enter the target loss")
                .validate_with(|input: &String| -> Result<(), &str> {
                    if let Ok(n) = input.parse::<f32>() {
                        if n >= 0. {
                            Ok(())
                        } else {
                            Err("Must be greater than 0.0")
                        }
                    } else {
                        Err("Must be a number")
                    }
                })
                .interact_text()
                .unwrap();
            Loss(l.parse::<f32>().unwrap())
        }
        _ => panic!(),
    };

    println!("{}", style("Begin training...").bold());
    println!("\tBatch size: {}", style(BATCH_SIZE).green());
    println!("\tLearning rate: {}\n", style(LEARNING_RATE).green());
    let start = Instant::now();
    network.train::<BATCH_SIZE, _>(dataset, LEARNING_RATE, train_stop_condition, &mut rng);
    let duration = start.elapsed();
    println!("Time elapsed for training the model: {:?}", style(duration).green());
}

fn convert_minmax_results_to_dataset(
    mut map: HashMap<TicTacToeState, SMatrix<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>>,
) -> (Vec<SVector<f32, TICTACTOE_GRID_SIZE>>, Vec<SVector<f32, TICTACTOE_GRID_SIZE>>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
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

        xs.push(x);
        ys.push(y);
    }

    (xs, ys)
}
