"""
Streaming DMD
A Python implementation of the streaming dynamic mode decomposition algorithm
described in the paper Liew, J. et al. "Streaming dynamic mode decomposition for
short-term forecasting in wind farms"

The algorithm performs a continuously updating dynamic mode decomposition as new
data is made available.

The equations referred to in this paper correspond to the equations in:
    Liew, J. et al. "Streaming dynamic mode decomposition for
    short-term forecasting in wind farms"

Author: Jaime Liew
License: MIT (see LICENSE.txt)
Version: 1.0
Email: jyli@dtu.dk
"""
from enum import Enum
import os
import matplotlib.pyplot as plt

import numpy as np

from .utilities import Delayer, Stacker


class Status(Enum):
    NONE = 0
    AUGMENTATION = 1
    REDUCTION = -1


def hankel_transform(X, s):
    """
    stacks the snapshots, X, so that each new snapshot contains the previous
    s snapshots.
    args:
        X (2D array): n by m matrix.
        s (int): stack size
    returns:
        Xout (2D array): n - (k-1) by k*m matrix.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if s == 1:
        return X
    l, m = X.shape
    w = m - (s - 1)
    out = np.zeros([l * s, w])

    for i in range(s):
        row = X[:, m - i - w : m - i]
        out[i * l : (i + 1) * l, :] = row

    return out


def truncatedSVD(X, r):
    """
    Computes the truncated singular value decomposition (SVD)
    args:
        X (2d array): Matrix to perform SVD on.
        rank (int or float): rank parameter of the svd. If a positive integer,
        truncates to the largest r singular values. If a float such that 0 < r < 1,
        the rank is the number of singular values needed to reach the energy
        specified in r. If -1, no truncation is performed.
    """

    U, S, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T
    if r >= 1:
        rank = min(r, U.shape[1])

    elif 0 < r < 1:
        cumulative_energy = np.cumsum(S**2 / np.sum(S**2))
        rank = np.searchsorted(cumulative_energy, r) + 1

    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = V[:, :rank]

    return U_r, S_r, V_r


class sDMD_base(object):
    """
    Calculate DMD in streaming mode.
    """

    def __init__(self, X, Y, rmin, rmax, thres=0.2, halflife=None):

        self.rmin = rmin
        self.rmax = rmax
        self.thres = thres
        self.halflife = halflife
        self.rho = 1 if halflife is None else 2 ** (-1 / halflife)

        # Eq. (2) - truncated SVD
        self.Ux, _, _ = truncatedSVD(X, rmin)
        self.Uy, _, _ = truncatedSVD(Y, rmin)

        # Eq. (3) - Mapping of input vector to reduced order space.
        X_tild = self.Ux.T @ X
        # Eq. (4) - Mapping of out vector to reduced order space.
        Y_tild = self.Uy.T @ Y

        # Eq (9) - Decomposition of transition matrix into the product of Q and Pinvx.
        self.Q = Y_tild @ X_tild.T
        self.Pinvx = X_tild @ X_tild.T
        self.Pinvy = Y_tild @ Y_tild.T

    def update(self, x, y):
        x, y = x.reshape([-1, 1]), y.reshape([-1, 1])
        status = 0

        normx = np.linalg.norm(x, ord=2, axis=0)
        normy = np.linalg.norm(y, ord=2, axis=0)

        xtilde = self.Ux.T @ x
        ytilde = self.Uy.T @ y

        # Numerator of Eq. (14) - projection error.
        ex = x - self.Ux @ xtilde
        ey = y - self.Uy @ ytilde

        x_status = Status.NONE
        y_status = Status.NONE
        #### STEP 1 - BASIS EXPANSION ####
        # Table 1: Rank augmentation of Ux
        if np.linalg.norm(ex, ord=2, axis=0) / normx > self.thres:

            u_new = ex / np.linalg.norm(ex, ord=2, axis=0)
            self.Ux = np.hstack([self.Ux, (u_new).reshape([-1, 1])])

            self.Pinvx = np.hstack([self.Pinvx, np.zeros([self.Pinvx.shape[0], 1])])
            self.Pinvx = np.vstack([self.Pinvx, np.zeros([1, self.Pinvx.shape[1]])])
            self.Q = np.hstack([self.Q, np.zeros([self.Q.shape[0], 1])])
            x_status = Status.AUGMENTATION

        # Table 1: Rank augmentation of Uy
        if np.linalg.norm(ey, ord=2, axis=0) / normy > self.thres:
            u_new = ey / np.linalg.norm(ey, ord=2, axis=0)
            self.Uy = np.hstack([self.Uy, (u_new).reshape([-1, 1])])

            self.Pinvy = np.hstack([self.Pinvy, np.zeros([self.Pinvy.shape[0], 1])])
            self.Pinvy = np.vstack([self.Pinvy, np.zeros([1, self.Pinvy.shape[1]])])
            self.Q = np.vstack([self.Q, np.zeros([1, self.Q.shape[1]])])
            y_status = Status.AUGMENTATION

        #### STEP 2 - BASIS POD COMPRESSION ####
        # Table 1: Rank reduction of Ux
        if self.Ux.shape[1] > self.rmax:
            eigval, eigvec = np.linalg.eig(self.Pinvx)
            indx = np.argsort(-eigval)
            eigval = eigval[indx]
            qx = eigvec[:, indx[: self.rmin]]

            self.Ux = self.Ux @ qx
            self.Q = self.Q @ qx
            self.Pinvx = np.diag(eigval[: self.rmin])
            x_status = Status.REDUCTION

        # Table 1: Rank reduction of Uy
        if self.Uy.shape[1] > self.rmax:
            eigval, eigvec = np.linalg.eig(self.Pinvy)
            indx = np.argsort(-eigval)
            eigval = -np.sort(-eigval)
            qy = eigvec[:, indx[: self.rmin]]

            self.Uy = self.Uy @ qy
            self.Q = qy.T @ self.Q
            self.Pinvy = np.diag(eigval[: self.rmin])
            y_status = Status.REDUCTION

        #### STEP 3 - REGRESSION UPDATE ####
        xtilde = self.Ux.T @ x
        ytilde = self.Uy.T @ y

        # Eq. (10), (11), and (12) - Rank 1 update of DMD matrices.
        self.Q = self.rho * self.Q + ytilde @ xtilde.T
        self.Pinvx = self.rho * self.Pinvx + xtilde @ xtilde.T
        self.Pinvy = self.rho * self.Pinvy + ytilde @ ytilde.T

        return x_status, y_status

    @property
    def rank(self):
        return self.Ux.shape[1] # TODO check: Changed from self.U.shape[1] to self.Ux.shape[1]

    @property
    def A(self):
        """
        Computes the reduced order transition matrix from xtilde to ytilde.
        """
        return self.Q @ np.linalg.pinv(self.Pinvx)

    @property
    def modes(self):
        """
        Compute DMD modes and eigenvalues. The first output is the eigenmode
        matrix where the columns are eigenvectors. The second output are the
        discrete time eigenvalues. Assumes the input and output space are the
        same.
        """

        eigvals, eigvecK = np.linalg.eig(self.Ux.T @ self.Uy @ self.A)
        modes = self.Ux @ eigvecK

        return modes, eigvals


class sDMD(sDMD_base):
    """
    A wrapper class for sDMD_base. Manages the streaming data inputs and
    transforms the data to correctly represent the additional channels and delay
    states as described in Section 2.3 - State augmentation.
    """

    def __init__(self, X, rmin, rmax, Y=None, f=1, s=1, **kwargs):
        self.s = s
        self.f = f
        if Y is None:
            Y = X
        self.rolling_x = X[:, -(s + f - 1) :]

        X_hank = hankel_transform(X, s)

        X_hank = X_hank[:, :-f]
        Y_init = Y[:, f + s - 1 :]

        super().__init__(X_hank, Y_init, rmin, rmax, **kwargs)

    def update(self, x_in, y_in=None):
        if y_in is None:
            y_in = x_in
        x_in = x_in.reshape(-1, 1)
        y_in = y_in.reshape(-1, 1)

        self.rolling_x = np.hstack([self.rolling_x, x_in])

        X_hank = hankel_transform(self.rolling_x, self.s)

        self.x_buff = X_hank[:, -1]
        xnew = X_hank[:, 0]
        ynew = y_in

        status = super().update(xnew, ynew)

        self.rolling_x = self.rolling_x[:, 1:]
        return status


class sDMDc_oneshot(sDMD_base):
    """
    An implementation of streaming dynamic mode decomposition with control.
    Produces a one-shot system y_k+1 = Ax_k + Bu_k.

    """

    def __init__(self, X, U, rmin, rmax, f=1, s=1, **kwargs):

        self.s = s
        self.f = f
        self.nu = U.shape[0]
        self.nx = X.shape[0]
        self.Xstack0 = Stacker(self.nx, self.s)
        self.Xstack1 = Delayer(self.nx * self.s, self.f)
        self.Ustack = Delayer(self.nu, self.f)

        X0_hank = np.hstack([self.Xstack0.update(x) for x in X.T])
        X_hank = np.hstack([self.Xstack1.update(x) for x in X0_hank.T])
        U_hank = np.hstack([self.Ustack.update(x) for x in U.T])

        XU_hank = np.vstack([X_hank, U_hank])

        Y = X[:, s + f :]
        XU_hank = XU_hank[:, s + f :]

        super().__init__(XU_hank, Y, rmin, rmax, **kwargs)

    def update(self, x_in, u_in):

        x_in = x_in.reshape(-1, 1)
        u_in = u_in.reshape(-1, 1)

        x0hank = self.Xstack0.update(x_in)
        xhank = self.Xstack1.update(x0hank)
        udel = self.Ustack.update(u_in)

        xnew = np.vstack([xhank, udel])
        y = x_in

        status = super().update(xnew, y)

        return status

    @property
    def B(self):
        return self.Uy @ self.A @ self.Ux[-self.nu :, :].T

    @property
    def CB(self):
        return self.B

    @property
    def modes(self):
        """
        Compute DMD modes and eigenvalues. The first output is the eigenmode
        matrix where the columns are eigenvectors. The second output are the
        discrete time eigenvalues. Assumes the input and output space are the
        same.
        """
        Ux1 = self.Ux[: -self.nu, :]
        eigvals, eigvecK = np.linalg.eig(Ux1.T @ self.Uy @ self.A)
        modes = Ux1 @ eigvecK

        return modes, eigvals


class sDMDc(sDMD_base):
    """
    An implementation of streaming dynamic mode decomposition with control.
    Produces a system x_k+1 = Ax_k + Bu_k.
    """

    def __init__(self, X, U, rmin, rmax, f=1, s=1, **kwargs):

        self.s = s
        self.f = f
        self.nu = U.shape[0]
        self.nx = X.shape[0]
        self.Xstack0 = Stacker(self.nx, self.s)
        self.Xstack1 = Delayer(self.nx * self.s, self.f)
        self.Ustack = Delayer(self.nu, self.f)

        Y_hank = np.hstack([self.Xstack0.update(x) for x in X.T])
        X_hank = np.hstack([self.Xstack1.update(x) for x in Y_hank.T])
        U_hank = np.hstack([self.Ustack.update(x) for x in U.T])

        XU_hank = np.vstack([X_hank, U_hank])

        Y_hank = Y_hank[:, s + f :]
        XU_hank = XU_hank[:, s + f :]

        super().__init__(XU_hank, Y_hank, rmin, rmax, **kwargs)

    def update(self, x_in, u_in):

        x_in = x_in.reshape(-1, 1)
        u_in = u_in.reshape(-1, 1)

        yhank = self.Xstack0.update(x_in)
        xhank = self.Xstack1.update(yhank)
        udel = self.Ustack.update(u_in)

        xnew = np.vstack([xhank, udel])

        status = super().update(xnew, yhank)

        return status

    @property
    def B(self):
        return self.Uy @ self.A @ self.Ux[-self.nu :, :].T

    @property
    def CB(self):
        return self.B[: self.nx, :]

    @property
    def modes(self):
        """
        Compute DMD modes and eigenvalues. The first output is the eigenmode
        matrix where the columns are eigenvectors. The second output are the
        discrete time eigenvalues. Assumes the input and output space are the
        same.
        """
        Ux1 = self.Ux[: -self.nu, :]
        eigvals, eigvecK = np.linalg.eig(Ux1.T @ self.Uy @ self.A)
        modes = Ux1 @ eigvecK

        return modes, eigvals

    def get_current_augmented_input_for_prediction(self):
        """
        Retrieves the current augmented input vector from internal state buffers.
        
        This method constructs the augmented input vector by combining the current
        delayed Hankelized states from Xstack1 and delayed control inputs from Ustack.
        The resulting vector represents the system state required for predicting the
        next time step using the learned DMD operator.
        
        Returns:
            np.ndarray: Augmented input vector of shape (nx*s + nu, 1) containing
                       stacked delayed state and control information.
                       
        Note:
            This method should be called before update() when predicting future states
            based on historical data up to the current time step k-1.
        """
        current_X_hank_delayed = self.Xstack1()
        current_U_delayed = self.Ustack()
        return np.vstack([current_X_hank_delayed, current_U_delayed])

    def predict_next_raw_state_from_augmented(self, current_augmented_input):
        """
        Predicts the next raw state vector using the learned DMD operator.
        
        This method applies the reduced-order DMD model to predict the next state
        given an augmented input vector containing delayed states and controls.
        The prediction follows the learned mapping from augmented input space to
        the next state in physical coordinates.
        
        Args:
            current_augmented_input (np.ndarray): Augmented input vector of shape
                                                 (nx*s + nu, 1) containing delayed
                                                 states and control inputs.
        
        Returns:
            np.ndarray: Predicted next raw state vector of shape (nx, 1), or
                       array of NaNs if prediction fails due to model unavailability
                       or dimensional incompatibility.
                       
        Raises:
            Warnings are printed for model unavailability or dimensional mismatches
            rather than raising exceptions to maintain robustness in streaming applications.
        """
        if self.A is None or self.Ux is None or self.Uy is None:
            return np.full((self.nx, 1), np.nan)
        
        if current_augmented_input.shape[0] != self.Ux.shape[0]:
            return np.full((self.nx, 1), np.nan)

        try:
            y_reduced_pred = self.A @ (self.Ux.T @ current_augmented_input)
            y_hank_physical_pred = self.Uy @ y_reduced_pred
            
            predicted_x_next_raw = y_hank_physical_pred[:self.nx, 0]
            return predicted_x_next_raw.reshape(-1, 1)
        except Exception:
            return np.full((self.nx, 1), np.nan)

    def predict_horizon(self, x_initial_raw_history, u_future_raw_sequence, num_predict_steps):
        """
        Performs multi-step ahead prediction using the learned sDMDc model.
        
        This method implements a rolling prediction scheme where the model iteratively
        predicts future states by feeding predicted outputs back as inputs for subsequent
        predictions. The method handles proper initialization of internal state buffers
        and maintains consistency with the training data structure.
        
        Args:
            x_initial_raw_history (np.ndarray): Historical state data of shape (nx, s+f-1)
                                               required to initialize internal stackers.
                                               The last column represents the current state x_k.
            u_future_raw_sequence (np.ndarray): Future control sequence of shape 
                                               (nu, num_predict_steps+f-1) containing
                                               historical controls for initialization
                                               followed by future control inputs.
            num_predict_steps (int): Number of future time steps to predict.
        
        Returns:
            np.ndarray: Predicted state trajectory of shape (nx, num_predict_steps)
                       containing the predicted evolution of the system state, or
                       None if model components are unavailable.
                       
        Raises:
            ValueError: If input dimensions are incompatible with model requirements.
            
        Note:
            The method creates temporary stacker objects to avoid modifying the
            internal state of the main model during prediction operations.
        """
        if self.A is None or self.Ux is None or self.Uy is None:
            print("Model (A, Ux, or Uy) not trained. Cannot perform multi-step prediction.")
            return None

        required_x_hist_len = self.s + self.f - 1
        if x_initial_raw_history.shape[0] != self.nx or x_initial_raw_history.shape[1] < required_x_hist_len:
            raise ValueError(
                f"x_initial_raw_history shape {x_initial_raw_history.shape} is invalid. "
                f"Expected ({self.nx}, >= {required_x_hist_len})."
            )
        
        required_u_len = num_predict_steps + self.f - 1
        if u_future_raw_sequence.shape[0] != self.nu or u_future_raw_sequence.shape[1] < required_u_len:
             raise ValueError(
                 f"u_future_raw_sequence shape {u_future_raw_sequence.shape} is invalid. "
                 f"Expected ({self.nu}, >= {required_u_len})."
            )

        predicted_X_raw_horizon = np.zeros((self.nx, num_predict_steps))
        
        pred_Xstack0 = Stacker(self.nx, self.s)
        pred_Xstack1 = Delayer(self.nx * self.s, self.f)
        pred_Ustack  = Delayer(self.nu, self.f)

        # Initialize prediction stackers with historical data
        for i in range(self.s):
            _ = pred_Xstack0.update(x_initial_raw_history[:, -(self.s-i)])
        
        temp_hist_x0_hank_for_Xstack1 = []
        temp_s0_for_priming = Stacker(self.nx, self.s)
        for i in range(required_x_hist_len):
            _ = temp_s0_for_priming.update(x_initial_raw_history[:, i])
            if i >= self.s - 1:
                temp_hist_x0_hank_for_Xstack1.append(temp_s0_for_priming())
        for i in range(self.f):
            _ = pred_Xstack1.update(temp_hist_x0_hank_for_Xstack1[-(self.f-i)])

        for i in range(self.f):
             _ = pred_Ustack.update(u_future_raw_sequence[:, i].reshape(-1,1))

        current_x_raw_for_loop = x_initial_raw_history[:, -1].reshape(-1,1)

        # Iterative prediction loop
        for i in range(num_predict_steps):
            current_u_raw_for_stacker = u_future_raw_sequence[:, self.f + i].reshape(-1,1)
            
            x0_hank = pred_Xstack0.update(current_x_raw_for_loop)
            x_hank_delayed = pred_Xstack1.update(x0_hank)
            u_delayed = pred_Ustack.update(current_u_raw_for_stacker)
            
            aug_input_for_step = np.vstack([x_hank_delayed, u_delayed])

            next_x_raw_pred = self.predict_next_raw_state_from_augmented(aug_input_for_step)
            
            if np.any(np.isnan(next_x_raw_pred)):
                print(f"Multi-step prediction encountered NaN at step {i+1}. Stopping.")
                return predicted_X_raw_horizon[:, :i]
            
            predicted_X_raw_horizon[:, i] = next_x_raw_pred.flatten()
            current_x_raw_for_loop = next_x_raw_pred
            
        return predicted_X_raw_horizon

    def plot_model_eigenvalues(self, A_true_for_comparison=None, 
                               save_path_prefix="plots/sdmdc_model_eigs", 
                               title_info=""):
        """
        Visualizes eigenvalues of the learned reduced-order operator.
        
        This method generates a complex plane plot showing the eigenvalues of the
        learned DMD operator, optionally comparing them with eigenvalues from a
        reference system matrix. The visualization includes the unit circle for
        stability assessment and handles multi-step prediction operators by
        showing appropriate power relationships.
        
        Args:
            A_true_for_comparison (np.ndarray, optional): Reference system matrix
                                                         for eigenvalue comparison.
            save_path_prefix (str, optional): File path prefix for saving the plot.
                                             Defaults to "plots/sdmdc_model_eigs".
            title_info (str, optional): Additional information for plot title.
        
        Note:
            The method automatically creates the output directory if it does not exist
            and handles numerical errors gracefully by printing appropriate warnings.
            For multi-step operators (f > 1), both original and power-scaled eigenvalues
            are displayed when a reference system is provided.
        """
        if self.A is None:
            print("Model operator A is None. Cannot plot eigenvalues.")
            return

        try:
            eigvals_learned_A_tilde = np.linalg.eigvals(self.A)

            plt.figure(figsize=(7, 7))
            plt.scatter(np.real(eigvals_learned_A_tilde), np.imag(eigvals_learned_A_tilde),
                        marker='o', s=80, facecolors='none', edgecolors='red', lw=1.5,
                        label=f'Learned A_tilde Evals (f_train={self.f})')

            if A_true_for_comparison is not None:
                eigvals_true_A = np.linalg.eigvals(A_true_for_comparison)
                plt.scatter(np.real(eigvals_true_A), np.imag(eigvals_true_A),
                            marker='x', s=100, color='black', lw=1.5,
                            label='True System A Evals (λ_true)')
                if self.f > 1:
                    eigvals_true_f_step_A = eigvals_true_A ** self.f
                    plt.scatter(np.real(eigvals_true_f_step_A), np.imag(eigvals_true_f_step_A),
                                marker='P', s=120, color='green', alpha=0.7,
                                label=f'True A Evals (λ_true^{self.f})')

            theta = np.linspace(0, 2 * np.pi, 100)
            plt.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.5, label='Unit Circle')

            plt.xlabel("Real Part")
            plt.ylabel("Imaginary Part")
            plot_title = f"sDMDc Model Eigenvalue Comparison"
            if title_info: plot_title += f" - {title_info}"
            plt.title(plot_title)
            plt.legend(fontsize=9)
            plt.axis('equal')
            plt.grid(True)
            
            plot_dir = os.path.dirname(save_path_prefix)
            if plot_dir and not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            filename_suffix = title_info.replace('=', '').replace('(', '').replace(')', '').replace(',', '_').replace(' ', '_').replace('/', '')
            filepath = f"{save_path_prefix}_{filename_suffix}.png"
            plt.savefig(filepath)
            plt.close()
            print(f"Saved model eigenvalues plot to {filepath}")

        except np.linalg.LinAlgError:
            print(f"Could not compute/plot eigenvalues (LinAlgError) for f_train={self.f}.")
        except Exception as e:
            print(f"Error plotting model eigenvalues: {e}")

    def get_effective_AB(self):
        """
        Extracts effective full-order system matrices from the learned reduced-order model.
        
        This method attempts to recover interpretable system matrices A_eff and B_eff
        that approximate the original system dynamics x_{k+1} = A_eff * x_k + B_eff * u_k.
        The extraction is most meaningful for models trained with minimal state stacking
        (s=1) and single-step prediction (f=1), where the learned operator directly
        approximates the system transition matrix.
        
        Returns:
            tuple: (A_eff, B_eff) where A_eff is the effective state transition matrix
                   of shape (nx, nx) and B_eff is the effective input matrix of shape
                   (nx, nu). Returns (None, None) if extraction conditions are not met
                   or if model components are unavailable.
                   
        Note:
            The method provides warnings when extraction conditions (s=1, f=1) are not
            strictly satisfied, as the interpretation of the extracted matrices becomes
            more complex for augmented state spaces and multi-step operators.
        """
        if not (self.s == 1 and self.f == 1):
            print("Warning: Effective A, B extraction is primarily for s=1, f=1 models.")

        if self.A is None or self.Ux is None or self.Uy is None:
            print("Model components (A, Ux, Uy) not available for A,B extraction.")
            return None, None

        try:
            Operator_learned_full = self.Uy @ self.A @ self.Ux.T
            A_B_block = Operator_learned_full[:self.nx, :] 

            expected_input_dim_for_AB_extraction = self.nx + self.nu
            if self.s == 1 and self.f == 1 and A_B_block.shape == (self.nx, expected_input_dim_for_AB_extraction):
                A_eff = A_B_block[:, :self.nx]
                B_eff = A_B_block[:, self.nx:]
                return A_eff, B_eff
            else:
                print(
                    f"Cannot directly extract A_eff, B_eff. Conditions s=1, f=1 might not be strictly met "
                    f"for this interpretation, or operator shape is unexpected. "
                    f"Operator_AB_block shape: {A_B_block.shape}, Expected: ({self.nx}, {expected_input_dim_for_AB_extraction}). "
                    f"Model s={self.s}, f={self.f}."
                )
                return None, None

        except Exception as e:
            print(f"Error during effective A, B extraction: {e}")
            return None, None
