use std::collections::HashMap;

use nalgebra::SMatrix;

use crate::tictactoe_game::TicTacToeCell::{Assigned, Empty};
use crate::tictactoe_game::TicTacToePlayer;
use crate::tictactoe_game::TicTacToePlayer::{O, X};
use crate::tictactoe_game::TicTacToeState;
use crate::TICTACTOE_SIZE;

/// Minmax solver for TicTacToe (X is the player of interest)
pub struct Minmax {
    cache: HashMap<TicTacToeState, i32>,
    result: HashMap<TicTacToeState, SMatrix<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>>,
}

impl Minmax {
    /// Executes the Minmax solver starting from a empty grid and
    /// `initial_player` as the first playing player and
    /// return a 'HashMap' that maps each possible turn of player X to the
    /// value of each action he can perform in that turn
    pub fn execute(initial_player: TicTacToePlayer) -> HashMap<TicTacToeState, SMatrix<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>> {
        let mut minmax = Minmax {
            cache: HashMap::new(),
            result: HashMap::new(),
        };

        minmax.minmax_rec(TicTacToeState::new(), initial_player);
        minmax.result
    }

    fn minmax_rec(&mut self, state: TicTacToeState, player: TicTacToePlayer) -> i32 {
        match state.winning_player() {
            Some(X) => return 1,
            Some(O) => return -1,
            _ => (),
        };

        let mut res = SMatrix::<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>::zeros();
        for i in 0..TICTACTOE_SIZE {
            for j in 0..TICTACTOE_SIZE {
                if state[i][j] == Empty {
                    let mut new_state = state;
                    new_state[i][j] = Assigned(player);

                    // use dynamic programming to avoid revisiting states
                    res[(i, j)] = match self.cache.get(&new_state) {
                        Some(i) => *i,
                        None => {
                            let res = self.minmax_rec(new_state, !player);
                            self.cache.insert(new_state, res);
                            res
                        }
                    }
                }
            }
        }

        match player {
            X => {
                self.result.insert(state, res);
                *res.into_iter().max().unwrap()
            }
            O => *res.into_iter().min().unwrap(),
        }
    }
}
