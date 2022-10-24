use nalgebra::{Const, SMatrix, SVector};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::{TICTACTOE_SIZE, TicTacToeCell, TicTacToePlayer, TicTacToeState};
use crate::TICTACTOE_GRID_SIZE;

pub struct TicTacToeNNSolver<const H: usize> {
    hidden_layer_weights: SMatrix<f32, H, TICTACTOE_GRID_SIZE>,
    hidden_layer_bias: SVector<f32, H>,
    output_layer_weights: SMatrix<f32, TICTACTOE_GRID_SIZE, H>,
    output_layer_bias: SVector<f32, TICTACTOE_GRID_SIZE>,
}

impl<const H: usize> TicTacToeNNSolver<H> {
    pub fn new<R: Rng + ?Sized>(r: &mut R) -> TicTacToeNNSolver<H> {
        // He initialization for ReLU and sigmoid neurons
        let relu_normal =
            Normal::new(0., (4. / (H + TICTACTOE_GRID_SIZE) as f32).sqrt()).unwrap();
        let sigmoid_normal =
            Normal::new(0., (32. / (H + TICTACTOE_GRID_SIZE) as f32).sqrt()).unwrap();

        TicTacToeNNSolver {
            hidden_layer_weights: SMatrix::from_fn(|_, _| relu_normal.sample(r)),
            hidden_layer_bias: SVector::from_fn(|_, _| relu_normal.sample(r)),
            output_layer_weights: SMatrix::from_fn(|_, _| sigmoid_normal.sample(r)),
            output_layer_bias: SVector::from_fn(|_, _| sigmoid_normal.sample(r)),
        }
    }

    pub fn inference(
        &self,
        state: TicTacToeState,
    ) -> SMatrix<bool, TICTACTOE_SIZE, TICTACTOE_SIZE> {
        let x = SMatrix::<f32, TICTACTOE_SIZE, TICTACTOE_SIZE>::from_fn(|i, j| match state[i][j] {
            TicTacToeCell::Empty => 0.,
            TicTacToeCell::Assigned(TicTacToePlayer::X) => 1.,
            TicTacToeCell::Assigned(TicTacToePlayer::O) => -1.,
        })
            .reshape_generic(Const::<TICTACTOE_GRID_SIZE>, Const::<1>);

        let y = self.forward::<1>(x);

        y.map(|y| y > 0.5)
            .reshape_generic(Const::<TICTACTOE_SIZE>, Const::<TICTACTOE_SIZE>)
    }

    fn forward<const BS: usize>(
        &self,
        x: SMatrix<f32, TICTACTOE_GRID_SIZE, BS>,
    ) -> SMatrix<f32, TICTACTOE_GRID_SIZE, BS> {
        // hidden layer
        //   dot product -> x1*wh1 + ... + x9*wh9
        let mut hidden_layer_output = self.hidden_layer_weights * x;
        //   add bias
        hidden_layer_output
            .column_iter_mut()
            .for_each(|mut c| c += self.hidden_layer_bias);
        //    apply relu
        hidden_layer_output.apply(|x| *x = ReLU(*x));

        // output layer
        //    dot product -> xh1*wo1 + ... + xhH*woH
        let mut output = self.output_layer_weights * hidden_layer_output;
        //    add bias
        output
            .column_iter_mut()
            .for_each(|mut c| c += self.output_layer_bias);
        //    apply sigmoid
        output.apply(|x| *x = Sigmoid(*x));

        output
    }
}

#[inline(always)]
fn ReLU(x: f32) -> f32 {
    let x = x.to_bits() as i32;
    f32::from_bits((x & !(x >> 31)) as u32)
}

#[inline(always)]
fn Sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
