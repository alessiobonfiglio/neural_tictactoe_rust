use std::collections::HashMap;
use std::ops::{Index, IndexMut, Not};

use nalgebra::SMatrix;

use crate::tictactoe_minmax::TicTacToeCell::{Assigned, Empty};
use crate::tictactoe_minmax::TicTacToePlayer::{O, X};

pub const TICTACTOE_SIZE: usize = 3;
pub const TICTACTOE_GRID_SIZE: usize = TICTACTOE_SIZE * TICTACTOE_SIZE;

/// Representation of a state of a TicTacToe game
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    state: [[TicTacToeCell; TICTACTOE_SIZE]; TICTACTOE_SIZE],
}

/// Representation of the TicTacToe players
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TicTacToePlayer {
    X,
    O,
}

/// Representation of a cell of a TicTacToe game's state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TicTacToeCell {
    Empty,
    Assigned(TicTacToePlayer),
}

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
    pub fn execute(
        initial_player: TicTacToePlayer,
    ) -> HashMap<TicTacToeState, SMatrix<i32, TICTACTOE_SIZE, TICTACTOE_SIZE>> {
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

impl TicTacToeState {
    /// Creates an empty state
    pub fn new() -> Self {
        TicTacToeState {
            state: [[Empty; 3]; 3],
        }
    }

    fn winning_player(&self) -> Option<TicTacToePlayer> {
        match self.state {
            [[Assigned(X), Assigned(X), Assigned(X)], _, _]
            | [_, [Assigned(X), Assigned(X), Assigned(X)], _]
            | [_, _, [Assigned(X), Assigned(X), Assigned(X)]]
            | [[Assigned(X), _, _], [Assigned(X), _, _], [Assigned(X), _, _]]
            | [[_, Assigned(X), _], [_, Assigned(X), _], [_, Assigned(X), _]]
            | [[_, _, Assigned(X)], [_, _, Assigned(X)], [_, _, Assigned(X)]]
            | [[Assigned(X), _, _], [_, Assigned(X), _], [_, _, Assigned(X)]]
            | [[_, _, Assigned(X)], [_, Assigned(X), _], [Assigned(X), _, _]] => Some(X),

            [[Assigned(O), Assigned(O), Assigned(O)], _, _]
            | [_, [Assigned(O), Assigned(O), Assigned(O)], _]
            | [_, _, [Assigned(O), Assigned(O), Assigned(O)]]
            | [[Assigned(O), _, _], [Assigned(O), _, _], [Assigned(O), _, _]]
            | [[_, Assigned(O), _], [_, Assigned(O), _], [_, Assigned(O), _]]
            | [[_, _, Assigned(O)], [_, _, Assigned(O)], [_, _, Assigned(O)]]
            | [[Assigned(O), _, _], [_, Assigned(O), _], [_, _, Assigned(O)]]
            | [[_, _, Assigned(O)], [_, Assigned(O), _], [Assigned(O), _, _]] => Some(O),

            _ => None,
        }
    }
}

impl Index<usize> for TicTacToeState {
    type Output = [TicTacToeCell; TICTACTOE_SIZE];

    fn index(&self, index: usize) -> &Self::Output {
        &self.state[index]
    }
}

impl IndexMut<usize> for TicTacToeState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.state[index]
    }
}

impl Not for TicTacToePlayer {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            X => O,
            O => X,
        }
    }
}
