use std::mem;

use console::{style, Term};
use nalgebra::{Const, SMatrix, SVector};
use rand::prelude::*;
use rand_distr::Normal;

use crate::{TICTACTOE_SIZE, TicTacToeCell, TicTacToePlayer, TicTacToeState};
use crate::TICTACTOE_GRID_SIZE;

pub struct TicTacToeSolverNN<const H: usize> {
    hidden_layer_weights: SMatrix<f32, H, TICTACTOE_GRID_SIZE>,
    hidden_layer_bias: SVector<f32, H>,
    output_layer_weights: SMatrix<f32, TICTACTOE_GRID_SIZE, H>,
    output_layer_bias: SVector<f32, TICTACTOE_GRID_SIZE>,
}

impl<const H: usize> TicTacToeSolverNN<H> {
    pub fn new<R: Rng + ?Sized>(r: &mut R) -> Self {
        // He initialization for re_lu and sigmoid neurons
        let relu_normal = Normal::new(0., (4. / (H + TICTACTOE_GRID_SIZE) as f32).sqrt()).unwrap();
        let sigmoid_normal = Normal::new(0., (32. / (H + TICTACTOE_GRID_SIZE) as f32).sqrt()).unwrap();

        TicTacToeSolverNN {
            hidden_layer_weights: SMatrix::from_fn(|_, _| relu_normal.sample(r)),
            hidden_layer_bias: SVector::from_fn(|_, _| relu_normal.sample(r)),
            output_layer_weights: SMatrix::from_fn(|_, _| sigmoid_normal.sample(r)),
            output_layer_bias: SVector::from_fn(|_, _| sigmoid_normal.sample(r)),
        }
    }

    pub fn get_parameters_size(&self) -> (usize, usize, usize, usize) {
        (
            mem::size_of_val(&self.hidden_layer_weights),
            mem::size_of_val(&self.hidden_layer_bias),
            mem::size_of_val(&self.output_layer_weights),
            mem::size_of_val(&self.output_layer_bias),
        )
    }

    pub fn inference(&self, state: TicTacToeState) -> SMatrix<bool, TICTACTOE_SIZE, TICTACTOE_SIZE> {
        let x = SMatrix::<f32, TICTACTOE_SIZE, TICTACTOE_SIZE>::from_fn(|i, j| match state[i][j] {
            TicTacToeCell::Empty => 0.,
            TicTacToeCell::Assigned(TicTacToePlayer::X) => 1.,
            TicTacToeCell::Assigned(TicTacToePlayer::O) => -1.,
        })
            .reshape_generic(Const::<TICTACTOE_GRID_SIZE>, Const::<1>);

        let (y, _) = self.forward::<1>(x);

        // instead of apply sigmoid, just check if y > 0 (same as sigmoid(y) > 0.5)
        y.map(|y| y > 0.).reshape_generic(Const::<TICTACTOE_SIZE>, Const::<TICTACTOE_SIZE>)
    }

    pub fn train<const BS: usize, R: Rng + ?Sized>(
        &mut self,
        dataset: (Vec<SVector<f32, TICTACTOE_GRID_SIZE>>, Vec<SVector<f32, TICTACTOE_GRID_SIZE>>),
        lr: f32,
        stop_condition: TrainStopCondition,
        r: &mut R,
    ) {
        let (mut x, mut y) = dataset;
        let ds_len = x.len();
        let mut optimizer = Adam::new(lr, 0.9, 0.999);

        let term = Term::stdout();
        term.write_line("").unwrap();

        let mut epoch_counter = 0;
        let steps_per_epoch = ds_len / BS;
        loop {
            epoch_counter += 1;
            Self::shuffle_dataset(&mut x, &mut y, r);
            let mut loss = 0.;
            let mut corrects = 0;
            for i in 0..steps_per_epoch {
                let start = i * BS;
                let end = start + BS;
                let xx = SMatrix::<f32, TICTACTOE_GRID_SIZE, BS>::from_columns(&x[start..end]);
                let yy = SMatrix::<f32, TICTACTOE_GRID_SIZE, BS>::from_columns(&y[start..end]);

                // forward pass
                let (out, hidden_layer_out) = self.forward(xx);
                loss += out.zip_map(&yy, bce_with_logits).mean();
                corrects += out.zip_map(&yy, |x, y| ((x > 0.) == (y > 0.5)) as i32).sum();

                // backward pass
                let loss_grad = out.zip_map(&yy, bce_with_logits_backward) / out.len() as f32;
                let grad = self.backward(loss_grad, hidden_layer_out, xx);

                optimizer.step(self, grad);
            }
            loss /= steps_per_epoch as f32;
            let accuracy = corrects as f32 / (9 * ds_len) as f32;
            term.clear_last_lines(1).unwrap();
            term.write_line(&format!("{}\t loss: {:.6}\t accuracy: {:.6}", style(format!("Epoch #{epoch_counter}")).bold(), loss, accuracy))
                .unwrap();

            if corrects as usize == 9 * ds_len || stop_condition.should_stop(epoch_counter, accuracy, loss) {
                break;
            }

            optimizer.reset();
        }
    }

    fn shuffle_dataset<R: Rng + ?Sized>(x: &mut [SVector<f32, TICTACTOE_GRID_SIZE>], y: &mut [SVector<f32, TICTACTOE_GRID_SIZE>], r: &mut R) {
        let ds_len = x.len();
        for i in 0..ds_len {
            let next = r.gen_range(i..ds_len);
            x.swap(i, next);
            y.swap(i, next);
        }
    }

    fn forward<const BS: usize>(&self, x: SMatrix<f32, TICTACTOE_GRID_SIZE, BS>) -> (SMatrix<f32, TICTACTOE_GRID_SIZE, BS>, SMatrix<f32, H, BS>) {
        // hidden layer
        //   dot product -> x1*wh1 + ... + x9*wh9
        let mut hidden_layer_output = self.hidden_layer_weights * x;
        //   add bias
        hidden_layer_output.column_iter_mut().for_each(|mut c| c += self.hidden_layer_bias);
        //    apply relu
        hidden_layer_output.apply(|x| *x = re_lu(*x));

        // output layer
        //    dot product -> xh1*wo1 + ... + xhH*woH
        let mut output = self.output_layer_weights * hidden_layer_output;
        //    add bias
        output.column_iter_mut().for_each(|mut c| c += self.output_layer_bias);

        (output, hidden_layer_output)
    }

    fn backward<const BS: usize>(
        &mut self,
        loss_grad: SMatrix<f32, TICTACTOE_GRID_SIZE, BS>,
        hidden_layer_out: SMatrix<f32, H, BS>,
        x: SMatrix<f32, TICTACTOE_GRID_SIZE, BS>,
    ) -> Self {
        let output_layer_bias_grad = loss_grad.column_mean();
        let output_layer_weights_grad = (loss_grad * hidden_layer_out.transpose()) / BS as f32;

        // compute gradient of the hidden layer output passing through the re_lu function
        let hidden_layer_output_grad = self
            .output_layer_weights
            .tr_mul(&loss_grad)
            .zip_map(&hidden_layer_out, |grad, x| grad * (x > 0.) as i32 as f32);

        let hidden_layer_bias_grad = hidden_layer_output_grad.column_mean();
        let hidden_layer_weights_grad = (hidden_layer_output_grad * x.transpose()) / BS as f32;

        TicTacToeSolverNN {
            hidden_layer_weights: hidden_layer_weights_grad,
            hidden_layer_bias: hidden_layer_bias_grad,
            output_layer_weights: output_layer_weights_grad,
            output_layer_bias: output_layer_bias_grad,
        }
    }
}

pub enum TrainStopCondition {
    Perfect,
    Accuracy(f32),
    Loss(f32),
    Epoch(u32),
}

impl TrainStopCondition {
    fn should_stop(&self, epoch: u32, accuracy: f32, loss: f32) -> bool {
        match self {
            TrainStopCondition::Perfect => false,
            TrainStopCondition::Accuracy(a) => accuracy >= *a,
            TrainStopCondition::Loss(l) => loss <= *l,
            TrainStopCondition::Epoch(e) => epoch >= *e,
        }
    }
}

#[inline(always)]
fn re_lu(x: f32) -> f32 {
    let x = x.to_bits() as i32;
    f32::from_bits((x & !(x >> 31)) as u32)
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

#[inline(always)]
fn bce_with_logits(x: f32, y: f32) -> f32 {
    let t = re_lu(-x);
    (1. - y) * x + t + ((-t).exp() + (-x - t).exp()).ln()
}

#[inline(always)]
fn bce_with_logits_backward(x: f32, y: f32) -> f32 {
    sigmoid(x) - y
}

struct Adam<const H: usize> {
    lr: f32,
    beta1: f32,
    beta2: f32,
    m: TicTacToeSolverNN<H>,
    v: TicTacToeSolverNN<H>,
    beta1t: f32,
    beta2t: f32,
}

impl<const H: usize> Adam<H> {
    fn new(lr: f32, beta1: f32, beta2: f32) -> Self {
        Adam {
            lr,
            beta1,
            beta2,
            m: TicTacToeSolverNN {
                hidden_layer_weights: SMatrix::zeros(),
                hidden_layer_bias: SVector::zeros(),
                output_layer_weights: SMatrix::zeros(),
                output_layer_bias: SVector::zeros(),
            },
            v: TicTacToeSolverNN {
                hidden_layer_weights: SMatrix::zeros(),
                hidden_layer_bias: SVector::zeros(),
                output_layer_weights: SMatrix::zeros(),
                output_layer_bias: SVector::zeros(),
            },
            beta1t: 1.,
            beta2t: 1.,
        }
    }

    fn reset(&mut self) {
        self.m = TicTacToeSolverNN {
            hidden_layer_weights: SMatrix::zeros(),
            hidden_layer_bias: SVector::zeros(),
            output_layer_weights: SMatrix::zeros(),
            output_layer_bias: SVector::zeros(),
        };
        self.v = TicTacToeSolverNN {
            hidden_layer_weights: SMatrix::zeros(),
            hidden_layer_bias: SVector::zeros(),
            output_layer_weights: SMatrix::zeros(),
            output_layer_bias: SVector::zeros(),
        };
        self.beta1t = 1.;
        self.beta2t = 1.;
    }

    fn step(&mut self, nn: &mut TicTacToeSolverNN<H>, grad: TicTacToeSolverNN<H>) {
        self.m.hidden_layer_weights = self.beta1 * self.m.hidden_layer_weights + (1. - self.beta1) * grad.hidden_layer_weights;
        self.m.hidden_layer_bias = self.beta1 * self.m.hidden_layer_bias + (1. - self.beta1) * grad.hidden_layer_bias;
        self.m.output_layer_weights = self.beta1 * self.m.output_layer_weights + (1. - self.beta1) * grad.output_layer_weights;
        self.m.output_layer_bias = self.beta1 * self.m.output_layer_bias + (1. - self.beta1) * grad.output_layer_bias;

        self.v.hidden_layer_weights =
            self.beta2 * self.v.hidden_layer_weights + (1. - self.beta2) * (grad.hidden_layer_weights.component_mul(&grad.hidden_layer_weights));
        self.v.hidden_layer_bias = self.beta2 * self.v.hidden_layer_bias + (1. - self.beta2) * grad.hidden_layer_bias.component_mul(&grad.hidden_layer_bias);
        self.v.output_layer_weights = self.beta2 * self.v.output_layer_weights + (1. - self.beta2) * grad.output_layer_weights.component_mul(&grad.output_layer_weights);
        self.v.output_layer_bias = self.beta2 * self.v.output_layer_bias + (1. - self.beta2) * grad.output_layer_bias.component_mul(&grad.output_layer_bias);

        self.beta1t *= self.beta1;
        self.beta2t *= self.beta2;

        let mt_hidden_layer_weights = self.m.hidden_layer_weights / (1. - self.beta1t);
        let mt_hidden_layer_bias = self.m.hidden_layer_bias / (1. - self.beta1t);
        let mt_output_layer_weights = self.m.output_layer_weights / (1. - self.beta1t);
        let mt_output_layer_bias = self.m.output_layer_bias / (1. - self.beta1t);

        let vt_hidden_layer_weights = (self.v.hidden_layer_weights / (1. - self.beta2t)).map(|x| x.sqrt());
        let vt_hidden_layer_bias = (self.v.hidden_layer_bias / (1. - self.beta2t)).map(|x| x.sqrt());
        let vt_output_layer_weights = (self.v.output_layer_weights / (1. - self.beta2t)).map(|x| x.sqrt());
        let vt_output_layer_bias = (self.v.output_layer_bias / (1. - self.beta2t)).map(|x| x.sqrt());

        nn.hidden_layer_weights -= self.lr * mt_hidden_layer_weights.component_div(&vt_hidden_layer_weights.add_scalar(10f32.powi(-8)));
        nn.hidden_layer_bias -= self.lr * mt_hidden_layer_bias.component_div(&vt_hidden_layer_bias.add_scalar(10f32.powi(-8)));
        nn.output_layer_weights -= self.lr * mt_output_layer_weights.component_div(&vt_output_layer_weights.add_scalar(10f32.powi(-8)));
        nn.output_layer_bias -= self.lr * mt_output_layer_bias.component_div(&vt_output_layer_bias.add_scalar(10f32.powi(-8)));
    }
}
