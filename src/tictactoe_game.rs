use std::io::Write;
use std::ops::{Index, IndexMut, Not};

use console::{style, Key, StyledObject, Term};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::tictactoe_game::TicTacToeCell::{Assigned, Empty};
use crate::tictactoe_game::TicTacToePlayer::{O, X};
use crate::tictactoe_solver_nn::TicTacToeSolverNN;

pub const TICTACTOE_SIZE: usize = 3;
pub const TICTACTOE_GRID_SIZE: usize = TICTACTOE_SIZE * TICTACTOE_SIZE;

/// Representation of a state of a TicTacToe game
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    state: [[TicTacToeCell; TICTACTOE_SIZE]; TICTACTOE_SIZE],
}

/// Representation of a TicTacToe game
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TicTacToeGame {
    state: TicTacToeState,
    current_player: TicTacToePlayer,
    user_player: TicTacToePlayer,
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

impl TicTacToeGame {
    /// Creates a new game with an empty grid, the first playing player being X and the user playing as `player`
    pub fn new(player: TicTacToePlayer) -> Self {
        TicTacToeGame {
            state: TicTacToeState::new(),
            current_player: X,
            user_player: player,
        }
    }

    /// Plays the game interactively against the network `nn`
    pub fn play<const H: usize, R: Rng + ?Sized>(&mut self, nn: &TicTacToeSolverNN<H>, r: &mut R) {
        let mut term = Term::stdout();
        let mut cur_i = 0;
        let mut cur_j = 0;
        loop {
            // print the grid
            for i in 0..TICTACTOE_SIZE {
                for j in 0..TICTACTOE_SIZE - 1 {
                    term.write_fmt(format_args!("{}│", Self::get_cell_string_colored(self.state[i][j]))).unwrap();
                }
                if i != TICTACTOE_SIZE - 1 {
                    term.write_fmt(format_args!("{}\n─┼─┼─\n", Self::get_cell_string_colored(self.state[i][TICTACTOE_SIZE - 1])))
                        .unwrap();
                } else {
                    term.write_fmt(format_args!("{}\n", Self::get_cell_string_colored(self.state[i][TICTACTOE_SIZE - 1])))
                        .unwrap();
                }
            }

            if let Some(winner) = self.state.winning_player() {
                if winner == self.user_player {
                    println!("You {}!\n", style("won").green());
                } else {
                    println!("You {}!\n", style("lost").red());
                }
                break;
            }
            if self.state.is_full() {
                println!("It's a {}!\n", style("tie").yellow());
                break;
            }

            if self.current_player == self.user_player {
                term.show_cursor().unwrap();
                term.move_cursor_up(2 * TICTACTOE_SIZE - 1).unwrap();
                term.move_cursor_down(2 * cur_i).unwrap();
                term.move_cursor_right(2 * cur_j).unwrap();
                loop {
                    match term.read_key().unwrap() {
                        Key::ArrowLeft => {
                            if cur_j > 0 {
                                cur_j -= 1;
                                term.move_cursor_left(2).unwrap();
                            }
                        }
                        Key::ArrowRight => {
                            if cur_j < TICTACTOE_SIZE - 1 {
                                cur_j += 1;
                                term.move_cursor_right(2).unwrap();
                            }
                        }
                        Key::ArrowUp => {
                            if cur_i > 0 {
                                cur_i -= 1;
                                term.move_cursor_up(2).unwrap();
                            }
                        }
                        Key::ArrowDown => {
                            if cur_i < TICTACTOE_SIZE - 1 {
                                cur_i += 1;
                                term.move_cursor_down(2).unwrap();
                            }
                        }
                        Key::Enter => {
                            if self.state[cur_i][cur_j] == Empty {
                                self.state[cur_i][cur_j] = Assigned(self.current_player);
                                self.current_player = !self.current_player;
                                break;
                            }
                        }
                        _ => (),
                    }
                }
                term.move_cursor_left(2 * cur_j).unwrap();
                term.move_cursor_down(2 * (TICTACTOE_SIZE - 1 - cur_i) + 1).unwrap();
                term.hide_cursor().unwrap();
            } else {
                let pred = nn.inference(self.state.to_solver_state(self.user_player));
                let mut possible_actions = Vec::new();
                for i in 0..TICTACTOE_SIZE {
                    for j in 0..TICTACTOE_SIZE {
                        if pred[(i, j)] && self.state[i][j] == Empty {
                            possible_actions.push((i, j));
                        }
                    }
                }
                // if the network didn't predict any valid action, consider them all
                if possible_actions.is_empty() {
                    for i in 0..TICTACTOE_SIZE {
                        for j in 0..TICTACTOE_SIZE {
                            if self.state[i][j] == Empty {
                                possible_actions.push((i, j));
                            }
                        }
                    }
                }

                // choose a random action from the predicted ones
                let (i, j) = possible_actions.choose(r).unwrap();
                self.state[*i][*j] = Assigned(self.current_player);
                self.current_player = !self.current_player;
            }
            term.clear_last_lines(2 * TICTACTOE_SIZE - 1).unwrap();
        }
    }

    fn get_cell_string_colored(cell: TicTacToeCell) -> StyledObject<&'static str> {
        match cell {
            Empty => style(" "),
            Assigned(X) => style("X").blue(),
            Assigned(O) => style("O").red(),
        }
    }
}

impl TicTacToeState {
    /// Creates an empty state
    pub fn new() -> Self {
        TicTacToeState { state: [[Empty; 3]; 3] }
    }

    /// Returns `true` if the grid is full
    pub fn is_full(&self) -> bool {
        self.state.iter().flatten().all(|c| *c != Empty)
    }

    /// Returns the player that won the game or `None` if the is no winner in the state
    pub fn winning_player(&self) -> Option<TicTacToePlayer> {
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

    fn to_solver_state(&self, player: TicTacToePlayer) -> TicTacToeState {
        if player == O {
            *self
        } else {
            let mut ret = *self;
            ret.state.iter_mut().flatten().for_each(|c| {
                *c = match c {
                    Empty => Empty,
                    Assigned(p) => Assigned(!*p),
                }
            });
            ret
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
